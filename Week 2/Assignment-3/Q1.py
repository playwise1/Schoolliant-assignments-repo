import pandas as pd
import random

emp_ids = []
departments = []
ages = []
salaries = []
experiences = []
performances = []
attritions = []

dept_choices = ['IT', 'HR', 'Finance', 'Sales', 'Marketing']
attrition_choices = ['Yes', 'No', 'No', 'No'] 

for i in range(200):
    emp_ids.append(1001 + i)
    departments.append(random.choice(dept_choices))
    ages.append(random.randint(22, 60))
    salaries.append(random.randint(40000, 150000))
    experiences.append(random.randint(1, 35))
    performances.append(random.randint(40, 100))
    attritions.append(random.choice(attrition_choices))

data = {
    'Employee_ID': emp_ids,
    'Department': departments,
    'Age': ages,
    'Salary': salaries,
    'Experience': experiences,
    'Performance': performances,
    'Attrition': attritions
}

df_setup = pd.DataFrame(data)
df_setup.to_csv('employee_dataset.csv', index=False)

df = pd.read_csv('employee_dataset.csv')

print("First 10 records:")
print(df.head(10))
print("\n")

avg_salary = df['Salary'].mean()
avg_age = df['Age'].mean()
avg_exp = df['Experience'].mean()

print("Average Salary:", avg_salary)
print("Average Age:", avg_age)
print("Average Experience:", avg_exp)
print("\n")

dept_counts = df['Department'].value_counts()
print("Employees in each department:")
print(dept_counts)
print("\n")

yes_attrition = df[df['Attrition'] == 'Yes']
dept_attrition_counts = yes_attrition['Department'].value_counts()
highest_attrition = dept_attrition_counts.index[0]

print("Department with highest attrition:")
print(highest_attrition)
print("\n")

filtered_employees = df[(df['Salary'] > 50000) & (df['Experience'] > 5)]
print("Filtered employees:")
print(filtered_employees)
print("\n")

sorted_employees = df.sort_values(by='Salary', ascending=False)
print("Sorted by salary:")
print(sorted_employees)
print("\n")

grouped_data = df.groupby('Department')[['Salary', 'Performance']].mean()
print("Grouped by Department averages:")
print(grouped_data)
print("\n")

def check_risk(row):
    if row['Attrition'] == 'Yes':
        return 'High Risk'
    elif row['Performance'] < 60:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['Risk_Level'] = df.apply(check_risk, axis=1)

print("Data with Risk Level:")
print(df.head(10))
print("\n")

df.to_csv('cleaned_employee_dataset.csv', index=False)
print("File exported successfully!")