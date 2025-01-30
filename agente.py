class AgentePedagogico:
    def __init__(self):
        self.nivel = "iniciante"

    def sugerir_conteudo(self):
        if self.nivel == "iniciante":
            return "Sugestão: Introdução à Matemática Básica."
        elif self.nivel == "intermediário":
            return "Sugestão: Álgebra e Geometria."
        else:
            return "Sugestão: Cálculo e Estatística Avançada."

    def atualizar_nivel(self, desempenho):
        if desempenho > 80:
            self.nivel = "avançado"
        elif desempenho > 50:
            self.nivel = "intermediário"
        else:
            self.nivel = "iniciante"


class AgenteAvaliador:
    def avaliar(self, respostas):
        total = len(respostas)
        corretas = sum(respostas)
        desempenho = (corretas / total) * 100
        return desempenho


class AgenteMotivacional:
    def motivar(self, desempenho):
        if desempenho > 80:
            return "Parabéns! Continue assim, você está indo muito bem!"
        elif desempenho > 50:
            return "Bom trabalho, mas você pode melhorar ainda mais!"
        else:
            return "Não desista! Pratique mais e você vai conseguir!"


class TutorInteligente:
    def __init__(self):
        self.agente_pedagogico = AgentePedagogico()
        self.agente_avaliador = AgenteAvaliador()
        self.agente_motivacional = AgenteMotivacional()

    def interagir(self, respostas):
        desempenho = self.agente_avaliador.avaliar(respostas)
        self.agente_pedagogico.atualizar_nivel(desempenho)
        conteudo = self.agente_pedagogico.sugerir_conteudo()
        motivacao = self.agente_motivacional.motivar(desempenho)
        return conteudo, motivacao


# Exemplo de uso
tutor = TutorInteligente()
respostas_do_aluno = [True, False, True, True, False]  # True = correta, False = incorreta
conteudo, motivacao = tutor.interagir(respostas_do_aluno)

print(conteudo)
print(motivacao)
