import numpy as np

def sphere_function(x):
    return np.sum(x**2)

class WOA:
    def __init__(self, func, dim, pop_size, max_iter, lb, ub):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.pos = np.random.uniform(lb, ub, (pop_size, dim))
        self.best_pos = np.zeros(dim)
        self.best_score = float('inf')
        self.convergence = []

    def check_boundary(self, X):
        return np.clip(X, self.lb, self.ub)

    def run(self):
        # Đánh giá ban đầu
        for i in range(self.pop_size):
            fitness = self.func(self.pos[i])
            if fitness < self.best_score:
                self.best_score = fitness
                self.best_pos = self.pos[i].copy()
        
        # Vòng lặp 
        for t in range(self.max_iter):
            a = 2 - 2 * t / self.max_iter
            
            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = np.random.uniform(-1, 1)
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) < 1: # Bao vây
                        D = abs(C * self.best_pos - self.pos[i])
                        self.pos[i] = self.best_pos - A * D
                    else: # Tìm kiếm ngẫu nhiên
                        rand_idx = np.random.randint(0, self.pop_size)
                        X_rand = self.pos[rand_idx]
                        D = abs(C * X_rand - self.pos[i])
                        self.pos[i] = X_rand - A * D
                else: # Xoắn ốc
                    dist = abs(self.best_pos - self.pos[i])
                    self.pos[i] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_pos
            
            # Kiểm tra biên và cập nhật
            for i in range(self.pop_size):
                self.pos[i] = self.check_boundary(self.pos[i])
                fitness = self.func(self.pos[i])
                if fitness < self.best_score:
                    self.best_score = fitness
                    self.best_pos = self.pos[i].copy()
            
            self.convergence.append(self.best_score)
            if t % 10 == 0:
                print(f"Iter {t}: Fitness = {self.best_score:.6f}")
                
        return self.best_score, self.best_pos

if __name__ == "__main__":
    print("=== DEMO WOA ===")
    algo = WOA(func=sphere_function, dim=30, pop_size=30, max_iter=100, lb=-100, ub=100)
    best_score, best_pos = algo.run()
    print(f"Kết quả: Best Fitness = {best_score}")