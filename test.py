from rag.chain import ask

result = ask("quelle est la posologie de la Terbinafine ?")

print(f"\n❓ Question : {result['question']}")
print(f"\n🤖 Réponse :\n{result['answer']}")
#print(f"\n📚 Sources utilisées :")
for s in result['sources']:
    print(f"  - {s['fichier']} | page {s['page']}")
    print(f"    {s['extrait']}\n")