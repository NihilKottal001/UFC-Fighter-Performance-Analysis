import numpy as np
import matplotlib.pyplot as plt


class FighterSkillComparison:
    def __init__(self, opponent1, opponent2) -> None:
        self.opponent1 = opponent1
        self.opponent2 = opponent2

    def fighter_skill_comparison(self, title):
        labels = self.opponent1.index
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        values1 = self.opponent1.tolist() + [self.opponent1[0]]
        ax.plot(angles, values1, color='blue', linewidth=2, linestyle='solid', label=self.opponent1.name)
        ax.fill(angles, values1, color='blue', alpha=0.25)
        values2 = self.opponent2.tolist() + [self.opponent2[0]]
        ax.plot(angles, values2, color='red', linewidth=2, linestyle='dashed', label=self.opponent2.name)
        ax.fill(angles, values2, color='red', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=13)
        ax.set_yticklabels([])
        ax.set_title(title, size=18, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.show()
        
        