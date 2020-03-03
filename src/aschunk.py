import numpy as np
from scipy.interpolate import interp1d
import pandas as pd


# A class to get the carious chunks given a relation
class SplineVisualChunker:


    def n_derivative(self,degree=1):
        """Compute the n-th derivative."""
        result = self.y.copy()
        for i in range(degree):
            result = np.gradient(result,self.x)
        return result

    def sign_change(self,arr):
        """Detect sign changes."""
        sign = np.sign(arr)
        result = ((np.roll(sign, 1) - sign) != 0).astype(bool)
        result[0] = False
        return result

    def zeroes(self,arr,d=1):
        """Find zeroes of an array.Sensitive to zero_threshold."""

        zero_threshold = self.zero_threshold * (10**d)

        #return self.sign_change(arr) | (abs(arr) < self.zero_threshold)
        return self.sign_change(arr)

    def find_critical(self):
        """Critical point, where first derivative is zero (stationary)"""
        return self.zeroes(self.d1)

    def find_maxima(self):
        return self.zeroes(self.d1) & (self.d2 < 0)

    def find_minima(self):
        return self.zeroes(self.d1) & (self.d2 > 0)

    def find_inflection(self):
        return self.zeroes(self.d2,d=2) & ~self.zeroes(self.d3,d=3)

    def find_saddle(self):
        return self.zeroes(self.d1,d=1) & self.zeroes(self.d2,d=2) & ~self.zeroes(self.d3,d=3)


    def evaluate(self, x, y, oversample=False, k=10, zero_threshold=1e-6):

        if oversample:
            f = interp1d(x, y, 'cubic')
            self.x = np.linspace(np.min(x),np.max(x),num=len(x)*k)
            self.y = f(self.x)
        else:
            self.x = x
            self.y = y

        self.zero_threshold = zero_threshold
        # Get derivatives
        self.d1 = self.n_derivative(degree=1)
        self.d2 = self.n_derivative(degree=2)
        self.d3 = self.n_derivative(degree=3)
        # Critical Point, first derivative is 0
        self.is_critical = self.find_critical()
        # Maxima,first derivative is 0 and second derivative < 0
        self.is_maxima = self.find_maxima()
        # Minima,first derivative is 0 and second derivative < 0
        self.is_minima = self.find_minima()
        self.is_inflection = self.find_inflection()
        self.is_saddle = self.find_saddle()
        self.num_critical = sum(self.is_critical)
        self.num_maxima = sum(self.is_maxima)
        self.num_minima= sum(self.is_minima)
        self.num_inflection = sum(self.is_inflection)
        self.num_saddle = sum(self.is_saddle)


    def get_plot_data(self, x_label, y_label):

        results = np.c_[self.x, self.y, self.is_critical, self.is_maxima, self.is_minima, self.is_inflection, self.is_saddle]

        df_results = pd.DataFrame(results,columns=[x_label,y_label,"is_critical","is_maxima","is_minima","is_inflection","is_saddle"])

        return df_results

    def get_num_chunks(self):

        return self.num_critical + 1


if __name__=="__main__":
    df = pd.read_csv("../Boston_Housing/boston_experiment1_pdp.csv")
    results = []
    for group_index, dfg in df.groupby(["path_index", "feature"]):
        path_index = group_index[0]
        feature = group_index[1]
        feature_values = df["feature_values"].loc[(df["path_index"] == path_index) & (df["feature"] == feature)].values
        prediction_outcome = df["feature_outcomes"].loc[(df["path_index"] == path_index) & (df["feature"] == feature)].values
        svc = SplineVisualChunker()
        evaluations = svc.evaluate(feature_values,prediction_outcome,oversample=False)
        evaluations=np.array([svc.x,svc.y,svc.is_critical,svc.is_minima,svc.is_maxima,svc.is_inflection,svc.is_saddle,svc.d1,svc.d2,svc.d3])
        evaluations=evaluations.transpose()
        df_result = pd.DataFrame(evaluations, columns=[feature, df.columns[1], "is_critical", "is_minima", "is_maxima", "is_inflection", "is_saddle","d1","d2","d3"])
        df_result.to_csv("../Boston_Housing/out_"+str(path_index)+"_"+feature+".csv",index=False)