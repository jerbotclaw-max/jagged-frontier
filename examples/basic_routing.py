"""
Example: Basic routing — see which model is best for a task.

Run: python examples/basic_routing.py
"""

from jagged import Router

router = Router()

tasks = [
    "Write a comedy sketch about two AI models arguing at a bar",
    "Implement a thread-safe LRU cache in Python",
    "Solve: what is the integral of sin(x) * cos(x)?",
    "Summarize the key themes of Moby Dick in 3 sentences",
    "Translate 'Knowledge is power' into French, German, and Japanese",
    "Analyze the pros and cons of microservices vs monolithic architecture",
    "What's happening in AI news right now?",
    "Describe this image: [photo of a sunset over mountains]",
]

print("=" * 70)
print("JAGGED FRONTIER — Basic Routing Examples")
print("=" * 70)

for task in tasks:
    decision = router.route(task)
    print(f"\n📋 Task:      {task}")
    print(f"   Type:      {decision.task_type} ({decision.confidence:.0%})")
    print(f"   ✅ Model:    {decision.selected_model} (score: {decision.score}/10)")
    runners_up = decision.alternatives[1:3]
    if runners_up:
        print(f"   ⚡ Alt:      {runners_up[0][0]} ({runners_up[0][1]:.1f}), {runners_up[1][0]} ({runners_up[1][1]:.1f})")
