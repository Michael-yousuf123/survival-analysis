from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(style='darkgrid', context='notebook',font_scale=1.5, color_codes=True)
import matplotlib.pyplot as plt
plt.style.use('ggplot')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
def compareplots(x, title = None):
    """Function that Plots boxplot and Heatmap correlation plots
    ============================================================
    ARGUMENTS: 
    x: Raw DataFrame
    title = String
    ============================================================"""
    fig, axes = plt.subplots(nrows= 1, ncols = 2, figsize = (18, 12), dpi = 80)
    # sns.pairplot(x,hue='Event',palette='gnuplot', ax=axes[0][0])
    sns.boxplot(data = x, palette = "Set1", ax=axes[0]).set(title='Data with Outliers')
    sns.heatmap(x.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', ax=axes[1])
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    plt.show()

def auc_plot(x = None, y = None):
    """Function to plot the AUC
    =============================
    ARGUMENTS:
    x: int or float ==> False Positive
    y: int or float ==> True Positive
    =============================
    """
    plt.figure(figsize=(20,10))
    plt.plot([0, 1], [0, 1], linestyle = "--")
    plt.plot(x, y, color="red", label = "Support Vector Machine") 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

def matrix_plot(x = None, y = None):
    """Function to plot Matrix
    ============================
    ARGUMENTS:
    x:(target label) int or float 
    y: (predicted label) int or float
    ============================"""
    class_names=[False, True] # name  of classes
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(confusion_matrix(x,y), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    tick_marks = [0.5, 1.5]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.show();