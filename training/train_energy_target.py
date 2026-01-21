# Policy improvement loop

# Action prior = mixture(VAE prior, policy)

# While true

# Train energy teacher

# Require:

# dynamics model

# action VAE prior

# trajectory dataset

# Roll out trajectory

# current precept = encode initial precept

# For each step

# If policy is not None

# action latent = generate action (policy improvement)

# else

# action latent = dataset action

# next precept = dynamics model (current precept, action latent)

# decomposition:

# S_t = f(X_t, S_t-1)

# X_t+1_pred = g(S_t, A_t)

# subsample k time steps in trajectory

# for each subsampled step

# for step in max(horizon, number of steps left in trajectory)

# sample n steps from mixture distribution

# calculate dynamics model(precept latent at subsampled step, action) for each sampled action

# for each result, compare with actual precept latent at subsampled step + 1

# calcuate NLL or MI

# Justification: was this situation already determined, or was it sensitive to the action i chose?

# average over n

# average over k

# Train energy student

# Use the created dataset to learn state -> discomfort mapping

# Train policy

# If policy == not trained (bootstrapping)

# Encode offline policy

# Back prop energy wrt actions

# Re-integrate

# Use updated actions as targets

# Update weights

# Update mixture

# If policy = trained (improvement)

# Roll out n trajectories

# Pick lowest energy

# Back prop E wrt energy

# Update targets

# Update weights

# Update mixture
