
f = open("/home/vinicius/Documentos/DeepReinforcement/LunarLander_OpenAI/Analises/Analise_3/output_files_008/others/test_report.txt", "r")
qtd = 0
for x in f:
	x = x.split("|")[1].split(":")[1]
	if float(x) >= 200:
		qtd += 1
print("A pontuação foi {}".format(qtd))
f.close()
