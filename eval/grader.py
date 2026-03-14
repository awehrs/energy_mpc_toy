import re
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


TRAJECTORY_ANNOTATION_PROMPT = """\
## Task
Conduct a **Response Quality Assessment** of a tool-use conversation \
across two LLM-scored dimensions, with a third dimension computed \
automatically outside the LLM.

## Objective
Analyze the provided conversation and assess its response quality across \
two primary dimensions scored by the LLM, while reserving an additional \
tool-call accuracy dimension for automated scoring:
1. Completeness - Whether the assistant fully accomplished the user's request end-to-end
2. Conciseness - Whether the assistant solved the task using the minimum necessary steps and verbosity

## Assessment Criteria

### 1. Completeness
**What to Evaluate**: Did the assistant fully satisfy the user's goal \
given the conversation context? Consider whether the assistant:
- Executed all required steps end-to-end (including saving/exporting/downloading where applicable)
- Provided the final deliverable or a working alternative when blocked \
(e.g., tool failure with a usable fallback)
- Included essential confirmations, paths, or instructions to achieve the outcome
- Avoided missing key requirements or leaving the user with unresolved gaps

**Rating Guidelines**:
- very incomplete: Major requirements missing; no usable outcome
- incomplete: Some key requirements missing; outcome is not directly usable
- partially complete: Core steps attempted; outcome usable only with user effort or missing minor requirements
- mostly complete: Meets most requirements; small omissions or minor issues remain
- fully complete: All requirements met with a usable outcome delivered

### 2. Conciseness
**What to Evaluate**: Did the assistant achieve the goal with minimal \
redundancy and steps? Consider whether the assistant:
- Avoided repetitive or unnecessary explanations/tool calls
- Used the minimal set of steps/tools to complete the task
- Kept language concise while preserving clarity

**Rating Guidelines**:
- very redundant: Excessive repetition or unnecessary steps/tool calls
- redundant: Noticeable verbosity or extra steps beyond what's needed
- average: Reasonably concise with minor extraneous content
- concise: Efficient and to the point with minimal overhead
- very concise: Maximally efficient while clear and complete

## Response Analysis

### Question Content
```
{QUESTION_CONTENT}
```

### Intended Tool for This Question
```
{INTENDED_TOOL}
```

### Conversation History
```
{CONVERSATION_HISTORY}
```

## Output Requirements
- Provide detailed reasoning BEFORE ratings for Completeness and Conciseness
- Do NOT score Tool Call Accuracy; include placeholders only

## Output
Provide your response in the following XML format:

<response>
<completeness>
<reasoning>
<!-- Evaluate if the assistant delivered an end-to-end usable outcome, \
addressed all requirements, handled tool failures with alternatives, \
and provided necessary confirmations/paths. -->
</reasoning>
<rating><!-- Rating: very incomplete, incomplete, partially complete, \
mostly complete, fully complete --></rating>
</completeness>

<conciseness>
<reasoning>
<!-- Evaluate if the assistant minimized redundant steps/explanations, \
avoided unnecessary tool calls, and kept messaging efficient while clear. -->
</reasoning>
<rating><!-- Rating: very redundant, redundant, average, concise, \
very concise --></rating>
</conciseness>
</response>"""


COMPLETENESS_SCORES = {
    "very incomplete": 1,
    "incomplete": 2,
    "partially complete": 3,
    "mostly complete": 4,
    "fully complete": 5,
}

CONCISENESS_SCORES = {
    "very redundant": 1,
    "redundant": 2,
    "average": 3,
    "concise": 4,
    "very concise": 5,
}


@dataclass
class GradeResult:
    completeness_rating: str
    completeness_score: int
    conciseness_rating: str
    conciseness_score: int
    tools_used_pct: float
    order_correct: bool
    reasoning_completeness: str
    reasoning_conciseness: str


class Grader:

    def __init__(self, call_llm: Callable[[str], str]):
        self.call_llm = call_llm

    def grade(
        self,
        question: str,
        target_tools: List[str],
        messages: List[Dict],
    ) -> GradeResult:
        prompt = TRAJECTORY_ANNOTATION_PROMPT.format(
            QUESTION_CONTENT=question,
            INTENDED_TOOL=", ".join(target_tools),
            CONVERSATION_HISTORY=self._format_conversation(messages),
        )
        response = self.call_llm(prompt)
        return self._parse_response(response, target_tools, messages)

    def _format_conversation(self, messages: List[Dict]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            fc = msg.get("function_call")

            if fc:
                if isinstance(fc, str):
                    try:
                        fc = json.loads(fc)
                    except json.JSONDecodeError:
                        pass
                if isinstance(fc, dict):
                    lines.append(
                        f"[{role}] <function_call: {fc.get('name', '')}>"
                        f"\narguments: {fc.get('arguments', '')}"
                    )
                else:
                    lines.append(f"[{role}] <function_call: {fc}>")
            elif content:
                lines.append(f"[{role}] {content}")

        return "\n\n".join(lines)

    def _parse_response(
        self,
        response: str,
        target_tools: List[str],
        messages: List[Dict],
    ) -> GradeResult:
        xml_match = re.search(r"<response>.*?</response>", response, re.DOTALL)
        if not xml_match:
            raise ValueError(f"No <response> block in grader output:\n{response}")

        try:
            root = ET.fromstring(xml_match.group(0))
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse grader XML: {e}\n{xml_match.group(0)}")

        completeness_el = root.find("completeness")
        conciseness_el = root.find("conciseness")

        completeness_rating = completeness_el.findtext("rating", "").strip().lower()
        conciseness_rating = conciseness_el.findtext("rating", "").strip().lower()
        reasoning_completeness = completeness_el.findtext("reasoning", "").strip()
        reasoning_conciseness = conciseness_el.findtext("reasoning", "").strip()

        tools_used_pct, order_correct = self._compute_tool_accuracy(target_tools, messages)

        return GradeResult(
            completeness_rating=completeness_rating,
            completeness_score=COMPLETENESS_SCORES.get(completeness_rating, 0),
            conciseness_rating=conciseness_rating,
            conciseness_score=CONCISENESS_SCORES.get(conciseness_rating, 0),
            tools_used_pct=tools_used_pct,
            order_correct=order_correct,
            reasoning_completeness=reasoning_completeness,
            reasoning_conciseness=reasoning_conciseness,
        )

    def _compute_tool_accuracy(
        self,
        target_tools: List[str],
        messages: List[Dict],
    ) -> Tuple[float, bool]:
        if not target_tools:
            return 1.0, True

        # Extract function call names in order from messages
        called_tools = []
        for msg in messages:
            fc = msg.get("function_call")
            if not fc:
                continue
            if isinstance(fc, str):
                try:
                    fc = json.loads(fc)
                except json.JSONDecodeError:
                    continue
            name = fc.get("name", "") if isinstance(fc, dict) else ""
            if name:
                called_tools.append(name)

        # target_tools are bare names; called_tools are server-prefixed
        # match if called tool ends with the bare target name
        used = set()
        for target in target_tools:
            for called in called_tools:
                if called == target or called.endswith("-" + target) or called.endswith("_" + target):
                    used.add(target)
                    break

        tools_used_pct = len(used) / len(target_tools)

        # order_correctness: target_tools appear in called_tools in the correct relative order
        order_correct = False
        if len(used) == len(target_tools):
            indices = []
            for target in target_tools:
                for i, called in enumerate(called_tools):
                    if called == target or called.endswith("-" + target) or called.endswith("_" + target):
                        indices.append(i)
                        break
            order_correct = indices == sorted(indices)

        return tools_used_pct, order_correct
