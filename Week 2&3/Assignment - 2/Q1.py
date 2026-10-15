import numpy as np

np.random.seed(42)

marks = np.random.randint(0, 101, size=(100, 5))
subjects = np.array(['Math', 'Physics', 'Chemistry', 'Biology', 'English'])
student_ids = np.arange(1, 101)

# 1. STUDENT ANALYTICS
total_marks = np.sum(marks, axis=1)
avg_marks = np.mean(marks, axis=1)

highest_total = np.max(total_marks)
lowest_total = np.min(total_marks)

top_student = student_ids[total_marks == highest_total][0]
lowest_student = student_ids[total_marks == lowest_total][0]

print("--- 1. STUDENT ANALYTICS ---")
print("Top Scorer: Student", top_student, "with", highest_total, "total marks.")
print("Lowest Scorer: Student", lowest_student, "with", lowest_total, "total marks.")
print("Total marks of first 5 students:", total_marks[0:5])
print("Average marks of first 5 students:", avg_marks[0:5])
print()

# 2. SUBJECT ANALYTICS
subject_averages = np.mean(marks, axis=0)
subject_highest = np.max(marks, axis=0)
subject_lowest = np.min(marks, axis=0)

print("--- 2. SUBJECT ANALYTICS ---")
for i in range(len(subjects)):
    print("Subject:", subjects[i])
    print("Avg:", subject_averages[i])
    print("High:", subject_highest[i])
    print("Low:", subject_lowest[i])
    print("-------------------")
print()

# 3. BOOLEAN MASKING
print("--- 3. BOOLEAN MASKING ---")
mask1 = avg_marks > 80
print("Students with >80 average:", student_ids[mask1])

mask2 = (marks[:, 0] < 35) | (marks[:, 1] < 35) | (marks[:, 2] < 35) | (marks[:, 3] < 35) | (marks[:, 4] < 35)
print("Students failing at least 1 subject:", student_ids[mask2])

mask3 = (marks[:, 0] > 90) | (marks[:, 1] > 90) | (marks[:, 2] > 90) | (marks[:, 3] > 90) | (marks[:, 4] > 90)
print("Students with >90 in ANY subject:", student_ids[mask3])
print()

# 4. RANKING SYSTEM
print("--- 4. RANKING SYSTEM ---")
sort_index = np.argsort(total_marks)

rank_1_idx = sort_index[-1]
rank_2_idx = sort_index[-2]
rank_3_idx = sort_index[-3]

print("Rank 1: Student", student_ids[rank_1_idx], "Marks:", total_marks[rank_1_idx])
print("Rank 2: Student", student_ids[rank_2_idx], "Marks:", total_marks[rank_2_idx])
print("Rank 3: Student", student_ids[rank_3_idx], "Marks:", total_marks[rank_3_idx])
print()

# 5. SCHOLARSHIP ELIGIBILITY
print("--- 5. SCHOLARSHIP ELIGIBILITY ---")
eligible_students = []

for i in range(100):
    student_avg = avg_marks[i]
    
    if student_avg >= 85:
        eligible_students.append(int(student_ids[i]))

if len(eligible_students) > 0:
    print("Eligible Students:", eligible_students)
else:
    print("No students met the strict scholarship criteria.")