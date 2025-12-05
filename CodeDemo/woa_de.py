import numpy as np
from woa import WOA

def sphere_function(x):
    return np.sum(x**2)

class WOA_DE(WOA):
    def run(self):
        for i in range(self.pop_size):
            fitness = self.func(self.pos[i])
            if fitness < self.best_score:
                self.best_score = fitness
                self.best_pos = self.pos[i].copy()
        
        for t in range(self.max_iter):
            a = 2 - 2 * t / self.max_iter
            
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p, l = np.random.rand(), np.random.uniform(-1, 1)
                
                if p < 0.5:
                    if abs(A) < 1:
                        self.pos[i] = self.best_pos - A * abs(C * self.best_pos - self.pos[i])
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        self.pos[i] = self.pos[rand_idx] - A * abs(C * self.pos[rand_idx] - self.pos[i])
                else:
                    self.pos[i] = abs(self.best_pos - self.pos[i]) * np.exp(l) * np.cos(2*np.pi*l) + self.best_pos
                
                self.pos[i] = self.check_boundary(self.pos[i])

            for i in range(self.pop_size):
                idxs = list(range(self.pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                
                # Mutation & Crossover
                mutant = self.best_pos + 0.5 * (self.pos[r1] - self.pos[r2])
                mutant = self.check_boundary(mutant)
                
                trial = np.zeros(self.dim)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() <= 0.9 or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = self.pos[i][j]
                
                if self.func(trial) < self.func(self.pos[i]):
                    self.pos[i] = trial
            
            for i in range(self.pop_size):
                fitness = self.func(self.pos[i])
                if fitness < self.best_score:
                    self.best_score = fitness
                    self.best_pos = self.pos[i].copy()
            
            if t % 10 == 0:
                print(f"Iter {t}: Fitness = {self.best_score:.6f}")

        return self.best_score, self.best_pos

if __name__ == "__main__":
    print("=== DEMO HYBRID WOA-DE ===")
    algo = WOA_DE(func=sphere_function, dim=30, pop_size=30, max_iter=100, lb=-100, ub=100)
    best_score, _ = algo.run()
    print(f"Kết quả: Best Fitness = {best_score}")