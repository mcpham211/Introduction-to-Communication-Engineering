import numpy as np
from woa import WOA 

def sphere_function(x):
    return np.sum(x**2)

class CWOA(WOA):
    def __init__(self, func, dim, pop_size, max_iter, lb, ub):
        super().__init__(func, dim, pop_size, max_iter, lb, ub)
        # Ghi đè khởi tạo
        self.pos = self.chaotic_initialization()

    def chaotic_initialization(self):
        chaos_pop = np.zeros((self.pop_size, self.dim))
        val = np.random.rand(self.dim)
        for i in range(self.pop_size):
            val = 4 * val * (1 - val) 
            chaos_pop[i] = self.lb + val * (self.ub - self.lb)
        return chaos_pop

if __name__ == "__main__":
    print("=== DEMO CHAOTIC WOA (CWOA) ===")
    algo = CWOA(func=sphere_function, dim=30, pop_size=30, max_iter=100, lb=-100, ub=100)
    best_score, _ = algo.run()
    print(f"Kết quả: Best Fitness = {best_score}")