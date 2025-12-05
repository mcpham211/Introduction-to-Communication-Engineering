import numpy as np
from woa import WOA

def sphere_function(x):
    return np.sum(x**2)

class OBL_WOA(WOA):
    def __init__(self, func, dim, pop_size, max_iter, lb, ub):
        super().__init__(func, dim, pop_size, max_iter, lb, ub)
        self.apply_obl()

    def apply_obl(self):
        # Tạo quần thể đối lập
        opp_pop = self.lb + self.ub - self.pos
        opp_pop = self.check_boundary(opp_pop)
        
        # Gộp và lọc
        combined = np.vstack((self.pos, opp_pop))
        fits = np.array([self.func(ind) for ind in combined])
        
        sorted_idx = np.argsort(fits)
        self.pos = combined[sorted_idx[:self.pop_size]]
        
        self.best_score = fits[sorted_idx[0]]
        self.best_pos = self.pos[0].copy()

if __name__ == "__main__":
    print("=== DEMO OBL-WOA ===")
    algo = OBL_WOA(func=sphere_function, dim=30, pop_size=30, max_iter=100, lb=-100, ub=100)
    best_score, _ = algo.run()
    print(f"Kết quả: Best Fitness = {best_score}")