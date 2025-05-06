"""
Collection of prompts with varying complexity for benchmarking the Phi-4-Mini-Reasoning model.
"""

# Short prompts
SHORT_PROMPTS = [
    {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "user", "content": "Tell me a joke."},
    {"role": "user", "content": "What's the weather like?"},
]

# Medium prompts
MEDIUM_PROMPTS = [
    {"role": "user", "content": "Explain the difference between machine learning and deep learning in a few sentences."},
    {"role": "user", "content": "What are the main features of Python programming language?"},
    {"role": "user", "content": "Summarize the plot of Romeo and Juliet."},
    {"role": "user", "content": "How does photosynthesis work?"},
    {"role": "user", "content": "What are the benefits and drawbacks of renewable energy?"},
]

# Long prompts
LONG_PROMPTS = [
    {"role": "user", "content": "Write a short essay about the impact of artificial intelligence on society in the next decade. Include both positive and negative aspects."},
    {"role": "user", "content": "Explain the theory of relativity in detail, including its historical context, key principles, and major implications for our understanding of physics."},
    {"role": "user", "content": "Compare and contrast three major programming paradigms: object-oriented, functional, and procedural. Provide examples of languages that primarily use each paradigm and discuss their strengths and weaknesses."},
    {"role": "user", "content": "Describe the process of machine learning from data collection to model deployment. Include explanations of data preprocessing, feature engineering, model selection, training, evaluation, and deployment considerations."},
    {"role": "user", "content": "Analyze the global climate change situation, including scientific evidence, potential impacts, international agreements, and proposed solutions. Discuss the challenges in implementing effective climate policies."},
]

# Reasoning prompts
REASONING_PROMPTS = [
    {"role": "user", "content": "If a train travels at 60 mph for 2 hours and then at 80 mph for 1 hour, what is the average speed for the entire journey?"},
    {"role": "user", "content": "A ball is thrown upward with an initial velocity of 20 m/s from a height of 1.5 meters. How high will the ball go? (Use g = 9.8 m/s²)"},
    {"role": "user", "content": "If 5 workers can build a wall in 8 days, how many days would it take for 10 workers to build the same wall, assuming all workers work at the same rate?"},
    {"role": "user", "content": "A store is offering a 20% discount on an item, and then applying a 10% tax. If the final price is $88, what was the original price before the discount and tax?"},
    {"role": "user", "content": "In a class of 30 students, 40% are boys. If 5 more girls join the class, what percentage of the class will be boys?"},
]

# Creative prompts
CREATIVE_PROMPTS = [
    {"role": "user", "content": "Write a short poem about the changing seasons."},
    {"role": "user", "content": "Create a brief story about a robot discovering emotions."},
    {"role": "user", "content": "Invent a new sport that combines elements of basketball and chess."},
    {"role": "user", "content": "Design a futuristic city that addresses current environmental challenges."},
    {"role": "user", "content": "Write a dialogue between the Sun and the Moon."},
]

# All prompts combined
ALL_PROMPTS = {
    "short": SHORT_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "long": LONG_PROMPTS,
    "reasoning": REASONING_PROMPTS,
    "creative": CREATIVE_PROMPTS,
}
