import matplotlib.pyplot as plt
from typing import List, Tuple, String, Any

def plot(items: List[Tuple[float, float, String]]) -> None:
	for item in items: 
		plt.plot(item[0], item[1], item[2])
