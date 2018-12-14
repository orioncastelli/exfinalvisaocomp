# exfinalvisaocomp
Exercício Final Visão Computacional


Como exercício final da cadeira de Visão Computacional na Pós Graduação de Inteligência Artificial 2018 - Universidade Positivo, foi desenvolvido pela equipe abaixo:

1. Bruno Henrique Hjort
2. Emanoel Kruger
3. Órion Rigel Castelli da Silva

O projeto consistia na captura do rosto de uma pessoa identificando o humor desta em três categorias, Feliz, Bravo e Neutro. Para chegarmos a tal resultado, utilizamos como modelo nosso colega Emanoel Kruger que submentou as três expressões a fim de gerarmos uma base de teste e treinamento para cada. 

Com as imagens capturadas rodamos uma rotina para identificação do rosto nas imagens utilizando o algoritmo Haar Cascade, extraindo o rosto da foto principal, a fim de busacrmos uma acurácia maior, em cima destas imagens geradas passamos um fitro para otimização e redimencionamento das imagens para 100x100.

Em cima das imagens extraídas e tratadas geramos a base de Teste e Treinamento para enfim gerar um modelo neural para predição. Este modelo foi gerado utilizando TensorFlow e a bilbioteca Keras, em cima deste modelo rodamos uma outra rotina utilizando o CV2 que em tempo real identifica o rosto em um vídeo, através do mesmo algoritimo Harr Cascade utilizado anteriormente, aplicando os filtros e redimensionamentos para então jogar o frame contra o modelo identificando o humor no exato momento da transmição.

Segue abaixo um descritivo sobre cada rotina desenvolvida na sequencia que deve ser rodada para a conclusão final.

Rotina 01: HumorArquivo.py
Remove rosto de uma imagem utilizando o HARR CASCADE, gerando uma nova imagem apenas com o rosto

Rotina 02: PreparaImagens.py
Responsável pelo tratamento das imagens a serem utilizadas no treinamento e teste da base, redimensionando estas para 100x100 e aplicando um filtro de cinza COLOR_BGR2GRAY

Rotina 03: Trabalho Treinamento.py
Treina modelo para predição do humor utilizando TensorFlow e Keras

Rotina 04: Trabalho.py
Através da biblioteca CV2 acessamos uma câmera para a geração em tempo real do humor do rosto na câmera, para cada frame no vídeo cruzamos contra o modelo gerado identificando o humor exibindo um rótulo acima do rosto identificado.

Incluímos também um vídeo demonstrativo do resultado final do exercício.
