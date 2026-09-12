from rag_pipeline import ask_agentic

question = input("Ask a question: ")
answer, debug = ask_agentic(question)

if debug["clarification_needed"]:
    print("\n--- CLARIFICATION NEEDED ---")
    print(answer)
else:
    print("\n--- ANSWER ---")
    print(answer)

print("\n--- DECISION TRAIL ---")
print(debug)