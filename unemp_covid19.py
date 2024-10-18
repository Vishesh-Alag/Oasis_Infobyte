import calendar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#preprocessed or modifying the data - 

df = pd.read_csv("unemployment_covid19.csv")
# Check for missing values
print(df.info())
print("\nChecking Missing Values : ")
print(df.isnull().sum()) 
# Check for duplicate rows
print("\nChecking Duplicated Rows : ")
print(df.duplicated().sum())
# Rename columns to remove leading/trailing spaces
df.columns = df.columns.str.strip()
# Rename 'Region' column to 'States'
df.rename(columns={'Region': 'States','Region.1':'Region'}, inplace=True)
# Display the updated column names to confirm the change
print(df.columns)
df['Date'] = df['Date'].str.strip()
# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
print(df.info())

# Converting 'Frequency' and 'Region' columns to categorical data type
df['Frequency'] = df['Frequency'].astype('category')
df['Region'] = df['Region'].astype('category')

# Extracting month from 'Date' and creating a 'Month' column
df['Month'] = df['Date'].dt.month

# Converting 'Month' to integer format
df['Month_int'] = df['Month'].apply(lambda x: int(x))

# Mapping integer month values to abbreviated month names
df['Month_name'] = df['Month_int'].apply(lambda x: calendar.month_abbr[x])

# Dropping the original 'Month' column
df.drop(columns='Month', inplace=True)

# Save the modified DataFrame to a new CSV file
#df.to_csv('modified_unemp_covid19.csv', index=False)
#print("Modified dataset saved as 'modified_unemp_covid19.csv'")'''

# Analysis will take place on Modified Data -----

df2 = pd.read_csv("modified_unemp_covid19.csv")
print("Information of Modified Unemployment Covid19 Data -- ")
print(df2.info())
print("\nDescription of Modified Unemployment Covid19 Data -- ")
df2_stat = df2[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']]
print(round(df2_stat.describe().T, 2))
print("\nFirst 5 Rows of Data -- ")
print(df2.head())

region_stats = df2.groupby(['Region'])[['Estimated Unemployment Rate (%)', 'Estimated Employed', 
                                       'Estimated Labour Participation Rate (%)']].mean().reset_index()
print(round(region_stats, 2))



#heatmap-
# Select relevant columns
m = df2[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)', 'longitude', 'latitude', 'Month_int']]
# Create a correlation matrix
hm = m.corr()
# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.set_context('notebook', font_scale=1)
sns.heatmap(data=hm, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), linewidths=0.5)
# Display the plot
plt.title("Correlation Heatmap")
plt.show()


# Plotting the unemployment rate over time for a specific state
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', hue='States', data=df2)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#boxplot to show unemployment rate accroding to states
fig = px.box(df2, x='States', y='Estimated Unemployment Rate (%)', color='States', title='Unemployment rate per States', template='seaborn')
# Updating the x-axis category order to be in descending total
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.show()

# Exploring the Relationships Between Employment Metrics: A Scatter Plot Matrix
fig = px.scatter_matrix(df2,template='seaborn',dimensions=['Estimated Unemployment Rate (%)', 'Estimated Employed',
                                                          'Estimated Labour Participation Rate (%)'],color='Region')
fig.show()


plot_unemp = df2[['Estimated Unemployment Rate (%)','States']]
df_unemployed = plot_unemp.groupby('States').mean().reset_index()

df_unemployed = df_unemployed.sort_values('Estimated Unemployment Rate (%)')

fig = px.bar(df_unemployed, x='States',y='Estimated Unemployment Rate (%)',color = 'States',title = 'Average unemployment rate in each state',
             template='seaborn')
fig.show()


fig = px.bar(df2, x='Region', y='Estimated Unemployment Rate (%)', animation_frame='Month_name', color='States',
             title='Unemployment rate across regions from Jan. 2020 to Oct. 2020', height=700, template='seaborn')

# Updating the x-axis category order to be in descending total
fig.update_layout(xaxis={'categoryorder': 'total descending'})

# Adjusting the animation frame duration
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 3000
fig.show()

# Creating a DataFrame with relevant columns
unemployed_df = df2[['States', 'Region', 'Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']]

unemployed = unemployed_df.groupby(['Region', 'States'])['Estimated Unemployment Rate (%)'].mean().reset_index()

# Creating a Sunburst chart - showing the unemployment rate in each Region and State
fig = px.sunburst(unemployed, path=['Region', 'States'], values='Estimated Unemployment Rate (%)', color_continuous_scale='lburdy',
                  title='Unemployment rate in each Region and State', height=550, template='presentation')

fig.show()



#Impact of Lockdown on States Estimated Employed using latitude and longitude

fig = px.scatter_geo(df2,'longitude', 'latitude', color="Region",
                     hover_name="States", size="Estimated Unemployment Rate (%)",
                     animation_frame="Month_name",scope='asia',template='seaborn',title='Impack of lockdown on Employement across regions')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 3000

fig.update_geos(lataxis_range=[5,35], lonaxis_range=[65, 100],oceancolor="#3399FF",
    showocean=True)

fig.show()


# Analysis based on before lockdown and after lockdown - 

# Filtering data for the period before the lockdown (January to April)
bf_lockdown = df2[(df2['Month_int'] >= 1) & (df2['Month_int'] <=4)]

# Filtering data for the lockdown period (April to July)
lockdown = df2[(df2['Month_int'] >= 4) & (df2['Month_int'] <=7)]

# Calculating the mean unemployment rate before lockdown by state
m_bf_lock = bf_lockdown.groupby('States')['Estimated Unemployment Rate (%)'].mean().reset_index()

# Calculating the mean unemployment rate after lockdown by state
m_lock = lockdown.groupby('States')['Estimated Unemployment Rate (%)'].mean().reset_index()

# Combining the mean unemployment rates before and after lockdown by state
m_lock['Unemployment Rate before lockdown (%)'] = m_bf_lock['Estimated Unemployment Rate (%)']

m_lock.columns = ['States','Unemployment Rate before lockdown (%)','Unemployment Rate after lockdown (%)']
print(m_lock)

#--------------------------------------

# percentage change in unemployment rate

m_lock['Percentage change in Unemployment'] = round(m_lock['Unemployment Rate after lockdown (%)'] - m_lock['Unemployment Rate before lockdown (%)']/m_lock['Unemployment Rate before lockdown (%)'],2)
plot_per = m_lock.sort_values('Percentage change in Unemployment')


# percentage change in unemployment after lockdown

fig = px.bar(plot_per, x='States',y='Percentage change in Unemployment',color='Percentage change in Unemployment',
            title='Percentage change in Unemployment in each state after lockdown',template='ggplot2',color_continuous_scale='Viridis')
fig.show()



'''The most affected states/territories in India during the lockdown in case of unemployment were:
Tripura
Haryana
Bihar
Puducherry
Jharkhand
Jammu & Kashmir
Delhi'''
