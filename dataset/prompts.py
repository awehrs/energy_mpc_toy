from typing import Dict, List, Optional, Union

import numpy as np

from tools.tools import (
    ToolSignature,
    GOOGLE_SIGNATURE,
    WIKIPEDIA_SIGNATURE,
)

# EXAMPLES.


Example = Dict[str, Union[str, List[Dict[str, Dict[str, str]]]]]

# Note, it took Claude Research 27 minutes to source these responses
HARD_EXAMPLES = [
    {
        "query": "What was the name of the chief architect who designed the Sydney Opera House, what other major building did they design that was never completed, and what specific engineering challenge caused that project to fail?",
        "steps": [
            {
                "calls": {
                    "google": "Sydney Opera House architect designer",
                    "wikipedia": "Sydney Opera House",
                },
                "responses": {
                    "google": "Danish architect Jørn Utzon won the international design competition for the Sydney Opera House in 1957. Born in 1918 in Copenhagen, he was 38 when he won the competition with design number 218.",
                    "wikipedia": "Designed by Danish architect Jørn Utzon and completed by an Australian architectural team headed by Peter Hall, the building was formally opened by Queen Elizabeth II on 20 October 1973, 16 years after Utzon's 1957 selection as winner of an international design competition.",
                },
            },
            {
                "calls": {
                    "google": "Jørn Utzon unfinished buildings projects never completed",
                    "wikipedia": "Jørn Utzon incomplete works",
                },
                "responses": {
                    "google": "Two of his projects had the earmarks of greatness: a theater in Zurich, and an intricate complex for the Kuwait National Assembly. Zurich, after nine years, was abandoned.",
                    "wikipedia": "Other noteworthy works include Bagsværd Church near Copenhagen and the National Assembly Building in Kuwait. The Schauspielhaus in Zurich was never completed.",
                },
            },
            {
                "calls": {
                    "google": "Schauspielhaus Zurich Utzon plywood beam structural system engineering failure",
                    "wikipedia": "Zurich Theatre construction challenges",
                },
                "responses": {
                    "google": "Utzon's revolutionary plywood structural system for the Sydney Opera House required £60,000 for prototypes, but funding was refused by the NSW Government after engineers questioned its viability, creating an impasse that led to Utzon's resignation and destroyed client confidence for future projects like Zurich.",
                    "wikipedia": "The complex design for a theatre in central Zurich was abandoned after it was instead decided to renovate the existing theatre. Utzon worked on the theatre for several years, but the design was abandoned when a decision was made to renovate the existing building.",
                },
            },
        ],
    },
    {
        "query": "Who was the PhD advisor of the person who invented the PageRank algorithm, what was that advisor's most cited paper, and which current AI technique traces its theoretical foundations to that paper?",
        "steps": [
            {
                "calls": {
                    "google": "PageRank inventor Larry Page Sergey Brin",
                    "wikipedia": "PageRank algorithm",
                },
                "responses": {
                    "google": "Larry Page and Sergey Brin developed PageRank at Stanford University in 1996 as part of a research project about a new kind of search engine.",
                    "wikipedia": "PageRank is an algorithm used by Google Search to rank web pages. It is named after both the term 'web page' and co-founder Larry Page. Larry Page and Sergey Brin developed PageRank at Stanford University in 1996.",
                },
            },
            {
                "calls": {
                    "google": "Terry Winograd Larry Page PhD advisor Stanford University",
                    "wikipedia": "Terry Winograd academic career",
                },
                "responses": {
                    "google": "His supervisor, Terry Winograd, encouraged him to pursue the idea, and Page recalled in 2008 that it was the best advice he had ever received. Sergey Brin's PhD advisor was Hector Garcia-Molina.",
                    "wikipedia": "Terry Winograd is an American professor of computer science at Stanford University, and co-director of the Stanford Human-Computer Interaction Group. He is known within the philosophy of mind and artificial intelligence fields for his work on natural language processing.",
                },
            },
            {
                "calls": {
                    "google": "Terry Winograd Understanding Computers and Cognition most cited paper",
                    "wikipedia": "Winograd Schema Challenge AI technique",
                },
                "responses": {
                    "google": "Terry Winograd's most cited paper 'Understanding Computers and Cognition' has 5,422 citations and led to the development of the Winograd Schema Challenge.",
                    "wikipedia": "The Winograd Schema Challenge directly influences modern transformer-based language models and conversational AI systems, serving as a benchmark for evaluating common sense reasoning in AI.",
                },
            },
        ],
    },
    {
        "query": "What was the first company acquired by Facebook, who was the founder of that company, and what startup is that founder currently working on?",
        "steps": [
            {
                "calls": {
                    "google": "Facebook first acquisition company purchased",
                    "wikipedia": "Facebook acquisitions",
                },
                "responses": {
                    "google": "Facebook's first acquisition was Parakey, announced on July 19, 2007. The acquisition brought the talents of Blake Ross and Joe Hewitt to Facebook.",
                    "wikipedia": "Parakey was a web application platform founded by Blake Ross and Joe Hewitt. It was acquired by Facebook in July 2007, making it Facebook's first acquisition.",
                },
            },
            {
                "calls": {
                    "google": "Blake Ross Joe Hewitt Parakey founders",
                    "wikipedia": "Blake Ross Joe Hewitt biography career",
                },
                "responses": {
                    "google": "Blake Ross and Joe Hewitt co-founded Parakey. Both were previously known as co-creators of the Firefox web browser at Mozilla.",
                    "wikipedia": "Blake Ross is an American software engineer best known for co-creating the Mozilla Firefox internet browser. Joe Hewitt is an American software programmer who co-created Firefox and later worked on Facebook's mobile applications.",
                },
            },
            {
                "calls": {
                    "google": "Blake Ross current startup 2024 recent ventures",
                    "wikipedia": "Blake Ross recent activities current projects",
                },
                "responses": {
                    "google": "Blake Ross's current activities are unclear since 2017. No recent startup ventures have been publicly announced.",
                    "wikipedia": "Joe Hewitt has completely transitioned to farming in Hawaii and is no longer involved in startup activities as of 2025.",
                },
            },
        ],
    },
    {
        "query": "What is the most downloaded app in India right now, which company owns it, and what was that company's stock price exactly one year ago today?",
        "steps": [
            {
                "calls": {
                    "google": "most popular downloaded app India 2024 statistics",
                    "wikipedia": "Mobile app usage India",
                },
                "responses": {
                    "google": "Instagram is India's most downloaded app in 2025 with 341 million downloads, followed by WhatsApp and other Meta-owned properties.",
                    "wikipedia": "India is one of the largest mobile app markets globally, with social media and messaging apps dominating download statistics.",
                },
            },
            {
                "calls": {
                    "google": "Instagram parent company owner developer",
                    "wikipedia": "Meta Platforms ownership",
                },
                "responses": {
                    "google": "Instagram is owned by Meta Platforms, Inc. (formerly Facebook, Inc.), which acquired Instagram in 2012 for $1 billion.",
                    "wikipedia": "Meta Platforms is an American multinational technology conglomerate that owns Facebook, Instagram, WhatsApp, and Threads.",
                },
            },
            {
                "calls": {
                    "google": "Meta stock price September 24 2024 historical data",
                    "wikipedia": "Meta Platforms stock market performance",
                },
                "responses": {
                    "google": "Unable to retrieve Meta's exact stock price for September 24, 2024, despite extensive searches across financial data sources.",
                    "wikipedia": "Meta Platforms trades on NASDAQ under the ticker symbol META, but specific historical daily prices require specialized financial data access.",
                },
            },
        ],
    },
    {
        "query": "What Nobel Prize was awarded in 2019 for Economics, who was the youngest recipient, and what was their undergraduate thesis topic?",
        "steps": [
            {
                "calls": {
                    "google": "Nobel Economics Prize 2019 winners recipients",
                    "wikipedia": "2019 Nobel Prize Economics",
                },
                "responses": {
                    "google": "The 2019 Nobel Prize in Economics was awarded to Abhijit Banerjee, Esther Duflo, and Michael Kremer for their experimental approach to alleviating global poverty.",
                    "wikipedia": "The Prize in Economic Sciences 2019 was awarded jointly to Abhijit Banerjee, Esther Duflo and Michael Kremer 'for their experimental approach to alleviating global poverty'.",
                },
            },
            {
                "calls": {
                    "google": "Esther Duflo Abhijit Banerjee Michael Kremer age youngest 2019",
                    "wikipedia": "Esther Duflo biography youngest Nobel winner",
                },
                "responses": {
                    "google": "Esther Duflo was the youngest recipient at age 46 when she won the Nobel Prize in 2019, making her the youngest person ever to win the Nobel Prize in Economics.",
                    "wikipedia": "Esther Duflo, born in 1972, was 46 years old when she received the Nobel Prize, making her both the youngest Nobel laureate in Economics and the second woman to receive the award.",
                },
            },
            {
                "calls": {
                    "google": "Esther Duflo undergraduate thesis topic Soviet propaganda",
                    "wikipedia": "Esther Duflo early academic work education history",
                },
                "responses": {
                    "google": "Esther Duflo's undergraduate thesis was on 'The use of propaganda during the Soviet Union's first five-year plan', examining how construction sites were represented in propaganda during Stalin's consolidation of power.",
                    "wikipedia": "Duflo's early academic work focused on economic history and development, including her undergraduate research on Soviet economic propaganda and its role in political consolidation.",
                },
            },
        ],
    },
    {
        "query": "Which university currently has the highest acceptance rate for international students, who is their current Dean of Admissions, and what admissions policy change did they implement in their previous role?",
        "steps": [
            {
                "calls": {
                    "google": "university highest acceptance rate international students 2024",
                    "wikipedia": "University admissions international students",
                },
                "responses": {
                    "google": r"Utah Valley University has the highest international student acceptance rate at effectively 100%, operating as an open enrollment institution.",
                    "wikipedia": "Open enrollment universities typically maintain very high acceptance rates, with some institutions accepting nearly all qualified applicants regardless of nationality.",
                },
            },
            {
                "calls": {
                    "google": "Utah Valley University Dean of Admissions current 2024",
                    "wikipedia": "Utah Valley University administrative leadership",
                },
                "responses": {
                    "google": "Utah Valley University's Director of Admissions is Kristopher 'Kris' Coles, who oversees undergraduate and international student admissions.",
                    "wikipedia": "Utah Valley University is part of the Utah System of Higher Education and maintains an open enrollment policy for most undergraduate programs.",
                },
            },
            {
                "calls": {
                    "google": "Kristopher Coles PRIME pilot program Utah System Higher Education",
                    "wikipedia": "Utah higher education policy changes admissions",
                },
                "responses": {
                    "google": "Kristopher Coles previously implemented the PRIME pilot program at Utah System of Higher Education, creating a three-tiered certificate system (LAUNCH, DISCOVER, TRANSFORM) that expanded concurrent enrollment access for K-12 students.",
                    "wikipedia": "The PRIME program established new pathways for high school students to access college-level coursework through an expanded concurrent enrollment system with multiple certification levels.",
                },
            },
        ],
    },
    {
        "query": "What was the last country to join NATO, what is their current defense spending as percentage of GDP, and does this meet NATO's target requirement?",
        "steps": [
            {
                "calls": {
                    "google": "most recent country joined NATO latest member",
                    "wikipedia": "NATO members",
                },
                "responses": {
                    "google": "Sweden was the last country to join NATO, officially becoming the 32nd member on March 7, 2024.",
                    "wikipedia": "Sweden joined NATO as the 32nd member state on 7 March 2024, following Finland which joined in 2023.",
                },
            },
            {
                "calls": {
                    "google": "Sweden defense spending GDP percentage 2024",
                    "wikipedia": "Sweden military expenditure",
                },
                "responses": {
                    "google": r"Sweden's current defense spending is 2.2% of GDP in 2024, projected to reach 2.4% in 2025 and 2.6% by 2028.",
                    "wikipedia": r"Sweden has significantly increased its defense budget in recent years, moving from below 1.5% of GDP to over 2% following geopolitical changes in Europe.",
                },
            },
            {
                "calls": {
                    "google": r"NATO defense spending requirement 2%% GDP Sweden compliance",
                    "wikipedia": "NATO defense spending targets member compliance",
                },
                "responses": {
                    "google": r"Yes, Sweden's 2.2% defense spending exceeds NATO's 2% GDP target requirement. Sweden is also planning to reach the new 3.5% target by 2030.",
                    "wikipedia": r"NATO's Wales Summit in 2014 established the guideline that member countries should spend at least 2% of GDP on defense. Sweden currently meets and exceeds this requirement.",
                },
            },
        ],
    },
]

EASY_EXAMPLES = []

EXAMPLES = HARD_EXAMPLES + EASY_EXAMPLES

TOOL_SIGNATURE_PLACE_HOLDER = "tool_signature_place_holder"

EXAMPLES_PLACE_HOLDER = "example_place_holder"

TOOLS_AVAILBLE_AT_STEP_N = "tools_available_at_step_n"

QUERY_PLACE_HOLDER = "query_place_holder"

STEP_PLACE_HOLDER = "step_place_holder"

PREVIOUS_CALLS_PLACE_HOLDER = "previous_calls_step_holder"

RESONSE_PLACE_HOLDER = "response_place_holder"

BASE_PROMPT = f"""
    Your task is to iteratively call one or more APIs to answer a query. 

    The names of the available API(s), along with their corresponding call formats are: 

    {TOOL_SIGNATURE_PLACE_HOLDER}

    Not every API will be available on every step. It's also possible that no APIs are called on 
    a given step. 

    This is a multistep task. After seeing the query, you will call the API's available at step 0. 
    Then, after seeing the API resonses, you will call the API's available at step 1, and so on.

    The API calls should help you incrementally acquire information to answer the initial query. 
    If the query seems difficult, you should initially make API calls that will help you acquire information
    that is useful for formulating better API calls at later steps, not for immediately answering the query. 
    If the query seems straightforward, however, you may want to make API calls intended to elicit that query's 
    answer directly. 

    Here examples of API calls and responses for hard questions. The first two turns of API calling and responses
    are shown along with some arbitrary later step "n":

    {EXAMPLES_PLACE_HOLDER}
    Now complete this sequence:

    Query: {QUERY_PLACE_HOLDER}
"""

CALL_PROMPT = f"""
    Step: {STEP_PLACE_HOLDER}
    Available APIs: {TOOLS_AVAILBLE_AT_STEP_N} 
    API calls: 
"""

RESPONSE_PROMPT = f"""
    {PREVIOUS_CALLS_PLACE_HOLDER}
    
    API responses: 

    {RESONSE_PLACE_HOLDER}
"""


class PromptBuilder:
    def __init__(
        self,
        tools: List[str],
        tool_signatures: Dict[str, ToolSignature],
        tool_probs: Dict[str, float],
        tool_examples: List[Example] = EXAMPLES,
        base_prompt: str = BASE_PROMPT,
        call_prompt: str = CALL_PROMPT,
        response_prompt: str = RESPONSE_PROMPT,
        num_examples: Optional[int] = None,
    ):
        self.tools = tools
        self.tool_signatures = tool_signatures
        self.tool_probs = tool_probs
        self.base_prompt = base_prompt
        self.call_prompt = call_prompt
        self.response_prompt = response_prompt

        # Build useful strings.
        self.tool_examples_str = self.build_example_string(tool_examples, num_examples)
        self.tool_signatures_str = self.build_tool_signature_string(tool_signatures)

    def build_example_string(
        self, tool_examples: List[Example], num_examples: Optional[int]
    ) -> str:
        if num_examples:
            tool_examples = tool_examples[:num_examples]

        examples = ""
        num_steps = len(tool_examples[0]["steps"])

        for idx, example in enumerate(tool_examples):
            query = example["query"]
            examples += f"\nExample {idx}\n{query}\n"

            for step in range(num_steps):
                calls = example["steps"][step]["calls"]
                responses = example["steps"][step]["responses"]

                available_tools = []

                for tool_name, prob in self.tool_probs.items():
                    if (
                        np.random.random() < prob
                    ):  # Include this tool with probability `prob`
                        available_tools.append(tool_name)

                current_calls = []
                current_responses = []

                if available_tools:
                    for tool in available_tools:
                        current_calls.append(
                            self.wrap_with_signature(
                                calls[tool], tool=tool, call_or_response="call"
                            )
                        )
                        current_responses.append(
                            self.wrap_with_signature(
                                responses[tool],
                                tool=tool,
                                call_or_response="response",
                            )
                        )

                step_num = "n" if step == num_steps - 1 else step
                if step_num == "n":
                    examples += "\n...[Additional steps]...\n"

                examples = self.add_step_prompt(examples, step_num, available_tools)
                examples = self.add_calls_and_responses(
                    examples, current_responses, current_calls
                )

        return examples

    def build_tool_signature_string(
        self, tool_signatures: Dict[str, ToolSignature]
    ) -> str:
        sig_strings = []

        for tool, sig in tool_signatures.items():
            tool_header = f"API: {tool}"
            call_header = f"Call format:"
            call_str = "".join([sig.call_start, "<call_text>", sig.call_end])
            resp_header = f"Response format:"
            resp_str = "".join(
                [sig.response_start, "<response_text>", sig.response_end]
            )
            sig_strings.append(
                "\n".join(
                    [
                        tool_header,
                        call_header,
                        call_str,
                        resp_header,
                        resp_str,
                    ]
                )
            )

        # Join all tool signatures with blank lines between them
        return "\n\n".join(sig_strings)

    def wrap_with_signature(
        self,
        text: str,
        tool: str,
        call_or_response: str,
    ) -> str:

        if call_or_response == "call":
            return (
                self.tool_signatures[tool].call_start
                + text
                + self.tool_signatures[tool].call_end
            )
        elif call_or_response == "response":
            return (
                self.tool_signatures[tool].response_start
                + text
                + self.tool_signatures[tool].response_end
            )
        else:
            raise ValueError("'call_or_response' must be one of 'call' or 'response'")

    @staticmethod
    def align_left(prompt: str) -> str:
        lines = prompt.split("\n")
        aligned_lines = [line.lstrip() for line in lines]
        return "\n".join(aligned_lines)

    def build_initial_prompt(self, query: str) -> str:
        """Build the initial prompt with tool signatures, examples, and query"""

        # Perform replacements.
        prompt = self.base_prompt.replace(
            TOOL_SIGNATURE_PLACE_HOLDER, self.tool_signatures_str
        )
        prompt = prompt.replace(EXAMPLES_PLACE_HOLDER, self.tool_examples_str)
        prompt = prompt.replace(QUERY_PLACE_HOLDER, query)

        # Align left
        return self.align_left(prompt)

    def add_step_prompt(
        self,
        current_prompt: str,
        step_num: Union[int, str],
        available_tools: List[str],
    ) -> str:
        """Add prompt for next step."""

        available_tools_text = ", ".join(available_tools) if available_tools else "None"

        call_section = self.call_prompt.replace(STEP_PLACE_HOLDER, str(step_num))

        call_section = call_section.replace(
            TOOLS_AVAILBLE_AT_STEP_N, available_tools_text
        )

        return current_prompt + self.align_left(call_section)

    def add_calls_and_responses(
        self, current_prompt: str, previous_calls: List[str], responses: List[str]
    ) -> str:
        """Add API calls and responses to step."""

        previous_calls_text = (
            "\n".join(previous_calls) if previous_calls else "No previous calls"
        )
        responses_text = "\n".join(responses) if responses else "No responses"

        response_section = self.response_prompt.replace(
            PREVIOUS_CALLS_PLACE_HOLDER, previous_calls_text
        )
        response_section = response_section.replace(
            RESONSE_PLACE_HOLDER, responses_text
        )

        return current_prompt + self.align_left(response_section)


PROMPT_DICT = {
    "google": None,
    "wikipedia": None,
}

if __name__ == "__main__":

    prompt_builder = PromptBuilder(
        tools=["google", "wikipedia"],
        tool_signatures={
            "google": GOOGLE_SIGNATURE,
            "wikipedia": WIKIPEDIA_SIGNATURE,
        },
        tool_examples=EXAMPLES,
        tool_probs={"google": 0.8, "wikipedia": 0.8},
        num_examples=2,
    )

    query = "What are the recent breakthroughs in CRISPR gene editing and their ethical implications?"

    initial_prompt = prompt_builder.build_initial_prompt(query)

    print(initial_prompt)

    assert False
    # Simulate some API calls and responses
    previous_calls = [
        "[GOOGLE_SEARCH]: recent CRISPR breakthroughs 2024",
        "[WIKIPEDIA_SEARCH]: CRISPR gene editing",
    ]

    responses = [
        "[GOOLE_RESPONSE] Found 3 recent papers on CRISPR improvements...",
        "[WIKIPEDIA_RESPONSE] is a gene-editing technology that...",
    ]

    # Add calls.
    step_0_prompt = prompt_builder.add_step_prompt(
        current_prompt=initial_prompt,
        step_num=0,
        available_tools=["google", "wikipedia"],
    )

    print(step_0_prompt)

    assert False
    # Add responses
    step_0_with_responses = prompt_builder.add_calls_and_responses(
        current_prompt=step_0_prompt,
        previous_calls=previous_calls,
        responses=responses,
    )

    print(step_0_with_responses)
