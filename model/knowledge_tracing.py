class KnowledgeTracing:
    def __init__(self, p_init, p_learn, p_guess, p_slip):
        self.p_L = p_init
        self.p_T = p_learn
        self.p_G = p_guess
        self.p_S = p_slip

    def update(self, correct):
        if correct == 1:
            numerator = self.p_L * (1 - self.p_S)
            denominator = numerator + (1 - self.p_L) * self.p_G
        else:
            numerator = self.p_L * self.p_S
            denominator = numerator + (1 - self.p_L) * (1 - self.p_G)

        self.p_L = numerator / denominator
        self.p_L = self.p_L + (1 - self.p_L) * self.p_T
        return self.p_L
