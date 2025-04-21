import os

folders = [
    "configs",
    "data/raw",
    "data/processed",
    "data/knowledge_graph",
    "experiments/retrieval",
    "experiments/verification",
    "models/retrieval",
    "models/verification",
    "models/shared",
    "notebooks",
    "outputs/runs",
    "outputs/predictions",
    "outputs/analysis",
    "src/data",
    "src/knowledge",
    "src/evaluation",
    "src/utils",
    "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    gitkeep_path = os.path.join(folder, ".gitkeep")
    with open(gitkeep_path, "w") as f:
        pass  # creates an empty .gitkeep file
    print(f"📁 Created: {folder} + .gitkeep")