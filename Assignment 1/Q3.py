"""Project 1 – Student Performance Analyzer
Create a Python program that:
Takes student name
Takes marks in 5 subjects
Calculates total
Calculates percentage
Assigns grade
Displays final report card
Concepts used:
Variables
Input
Functions
If-Else
Lists"""


def grade(percentage):
    if percentage >= 90:
        return "A"
    elif percentage >= 75:
        return "B"
    elif percentage >= 60:
        return "C"
    elif percentage >= 40:
        return "D"
    else:
        return "F"


name = input("Enter student name: ")

marks = []

for i in range(5):
    mark = int(input("Enter marks: "))
    marks.append(mark)

total = sum(marks)
percentage = total / 5
final_grade = grade(percentage)

print("\n--- Report Card ---")
print("Name:", name)
print("Marks:", marks)
print("Total:", total)
print("Percentage:", percentage, "%")
print("Grade:", final_grade)