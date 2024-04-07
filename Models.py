from ImportData import *

US_data = US_data[:-1]
countries = [US_data]
print(US_data)

for country_data in countries:
    # n_factors = len(US_factor_list)

    factor_names = country_data.columns.tolist()
    factors = country_data[factor_names]
    target = country_data['ER']

    #Plot correlations of factors
    sb.heatmap(factors.corr(), annot=True, cbar=False)
    plt.title('Correlation Matrix - All Factors')
    plt.show()

    sb.heatmap(factors.corr() > 0.9, annot=True, cbar=False)
    plt.title('Correlation Matrix - All Factors Above 0.9 Correlation')
    plt.show()

    sb.heatmap(factors.corr() < -0.9, annot=True, cbar=False)
    plt.title('Correlation Matrix - All Factors Below -0.9 Correlation')
    plt.show()

    print(country_data.describe())
    print(country_data.median())

    y = country_data.iloc[:,0]
    x = country_data.iloc[:,1:]
    x = sm.add_constant(x)

    model = sm.OLS(y,x).fit()
    print(model.summary())

