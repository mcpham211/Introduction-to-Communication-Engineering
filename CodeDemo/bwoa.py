import numpy as np

def sigmoid(x):
    exponent = -10 * (x - 0.5)
    exponent = np.clip(exponent, -500, 500) 
    return 1 / (1 + np.exp(exponent))

def onemax_function(x):
    return np.sum(x == 0) 

class BWOA:
    def __init__(self, dim, pop_size, max_iter):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pos = np.random.randint(0, 2, (pop_size, dim))
        self.X_cont = np.random.uniform(0, 1, (pop_size, dim))
        self.best_pos = np.zeros(dim)
        self.best_score = float('inf')

    def run(self):
        for i in range(self.pop_size):
            fit = onemax_function(self.pos[i])
            if fit < self.best_score:
                self.best_score = fit
                self.best_pos = self.pos[i].copy()

        for t in range(self.max_iter):
            a = 2 - 2 * t / self.max_iter
            
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p, l = np.random.rand(), np.random.uniform(-1, 1)
                
                new_pos_cont = np.zeros(self.dim)
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * self.best_pos - self.X_cont[i])
                        new_pos_cont = self.best_pos - A * D
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        new_pos_cont = self.X_cont[rand_idx] - A * abs(C * self.X_cont[rand_idx] - self.X_cont[i])
                else:
                    new_pos_cont = abs(self.best_pos - self.X_cont[i]) * np.exp(l) * np.cos(2*np.pi*l) + self.best_pos
                
                self.X_cont[i] = new_pos_cont
                
                # --- CHUYỂN ĐỔI ---
                T = sigmoid(new_pos_cont)
                self.pos[i] = np.where(np.random.rand(self.dim) < T, 1, 0)

            for i in range(self.pop_size):
                fit = onemax_function(self.pos[i])
                if fit < self.best_score:
                    self.best_score = fit
                    self.best_pos = self.pos[i].copy()
            
            if t % 10 == 0:
                print(f"Iter {t}: Số bit 0 còn lại = {self.best_score}")
                
        return self.best_score, self.best_pos

if __name__ == "__main__":
    print("=== DEMO BINARY WOA (BWOA) ===")
    algo = BWOA(dim=30, pop_size=30, max_iter=50)
    score, pos = algo.run()
    print(f"Kết quả: Bit 0 = {score}")