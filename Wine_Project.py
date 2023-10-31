import pandas as pd
import altair as alt
import numpy as np
wine_data= pd.read_csv("winequality-red.csv")
wine_data.head()
wine_data_clean = wine_data.dropna()

#Q: How do different levels of acidty (fixed acidity, volatile acidity, and pH) relate to wine quality?
# Scatter plot for 'Fixed Acidity' by 'Wine Quality'
fixed_acidity_scatter = alt.Chart(wine_data_clean).mark_circle().encode(
    x=alt.X('quality:O', title='Wine Quality'),
    y=alt.Y('fixed acidity:Q', title='Fixed Acidity'),
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,
    title='Fixed Acidity by Wine Quality'
)

fixed_acidity_scatter

# Scatter plot for 'Volatile Acidity' by 'Wine Quality'
volatile_acidity_scatter = alt.Chart(wine_data_clean).mark_circle().encode(
    x=alt.X('quality:O', title='Wine Quality'),
    y=alt.Y('volatile acidity:Q', title='Volatile Acidity'),
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,
    title='Volatile Acidity by Wine Quality'
)



# Scatter plot for 'pH' by 'Wine Quality'
pH_scatter = alt.Chart(wine_data_clean).mark_circle().encode(
    x=alt.X('quality:O', title='Wine Quality'),
    y=alt.Y('pH:Q', title='pH'),
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,
    title='pH by Wine Quality'
)

fixed_acidity_scatter


import altair as alt

# Scatter plot for 'Fixed Acidity' by 'Wine Quality' with swapped X and Y
fixed_acidity_scatter = alt.Chart(wine_data_clean).mark_circle().encode(
    y=alt.Y('quality:O', title='Wine Quality', scale=alt.Scale(reverse=True)),
    x=alt.X('fixed acidity:Q', title='Fixed Acidity'),
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,  # Adjust the width to make it horizontal
    title='Fixed Acidity by Wine Quality'
)

# Scatter plot for 'Volatile Acidity' by 'Wine Quality' with swapped X and Y
volatile_acidity_scatter = alt.Chart(wine_data_clean).mark_circle().encode(
    y=alt.Y('quality:O', title='Wine Quality', scale=alt.Scale(reverse=True)),
    x=alt.X('volatile acidity:Q', title='Volatile Acidity'),
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,  # Adjust the width to make it horizontal
    title='Volatile Acidity by Wine Quality'
)

# Scatter plot for 'pH' by 'Wine Quality' with swapped X and Y
pH_scatter = alt.Chart(wine_data_clean).mark_circle().encode(
    y=alt.Y('quality:O', title='Wine Quality', scale=alt.Scale(reverse=True)),
    x=alt.X('pH:Q', title='pH'),
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,  # Adjust the width to make it horizontal
    title='pH by Wine Quality'
)

# Combine the scatter plots in a row
scatter_plots = fixed_acidity_scatter | volatile_acidity_scatter | pH_scatter

scatter_plots


import altair as alt

# Create a faceted boxplot with X and Y axes flipped
faceted_boxplot = alt.Chart(wine_data_clean).mark_boxplot().encode(
    x=alt.X('fixed acidity:Q', title='Fixed Acidity'),
    y=alt.Y('quality:O', title='Wine Quality', sort='descending'),  # Reverse the order of 'quality'
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,
    title='Distribution of Fixed Acidity by Wine Quality'
).facet(
    column=alt.Column('attribute:N', title='Acidity Attribute')
)

faceted_boxplot |= alt.Chart(wine_data_clean).mark_boxplot().encode(
    x=alt.X('volatile acidity:Q', title='Volatile Acidity'),
    y=alt.Y('quality:O', title='Wine Quality', sort='descending'),  # Reverse the order of 'quality'
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,
    title='Distribution of Volatile Acidity by Wine Quality'
).facet(
    column=alt.Column('attribute:N', title='Acidity Attribute')
)

faceted_boxplot |= alt.Chart(wine_data_clean).mark_boxplot().encode(
    x=alt.X('pH:Q', title='pH'),
    y=alt.Y('quality:O', title='Wine Quality', sort='descending'),  # Reverse the order of 'quality'
    color=alt.Color('quality:N', title='Quality', legend=None)
).properties(
    width=200,
    title='Distribution of pH by Wine Quality'
).facet(
    column=alt.Column('attribute:N', title='Acidity Attribute')
)

faceted_boxplot.resolve_scale(y='independent')


#faceted_boxplot.save('interactive_faceted_boxplot.html', embed_options={'renderer':'svg'})






