# for now just find the differntial equations papers.
# Parse through the entire papers.
# Take the books and generate the required data using a big muscular math model. Or even better, find an oepn source ds with questions and anser


# just take a look at the clp quesitons and see whatsup.

import pandas as pd 
import matplotlib.pyplot as plt 
# 1806 quesitons 


def plot_category_bar_chart():
    df = pd.read_csv("CLP.csv")
    category_counts = df["category"].value_counts()
    category_counts.plot(kind='bar')
    plt.title('Distribution of Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=0) 
    plt.show()



