this dataset is taken from the source https://analyse.kmi.open.ac.uk/open-dataset

courses.csv

ALl sttributes understanding
code_module - code name of the module, which serves as the identifier.
code_presentation - code name of the presentation. It consists of the year and "B" for the presentation starting in February and "J" for the presentation starting in October.
length - length of the module-presentation in days.
The structure of B and J presentations may differ and therefore it is good practice to analyse the B and J presentations separately. Nevertheless, for some presentations the corresponding previous B/J presentation do not exist and therefore the J presentation must be used to inform the B presentation or vice versa. In the dataset this is the case of CCC, EEE and GGG modules.

assessments.csv
This file contains information about assessments in module-presentations. Usually, every presentation has a number of assessments followed by the final exam. CSV contains columns:

code_module - identification code of the module, to which the assessment belongs.
code_presentation - identification code of the presentation, to which the assessment belongs.
id_assessment - identification number of the assessment.
assessment_type - type of assessment. Three types of assessments exist: Tutor Marked Assessment (TMA), Computer Marked Assessment (CMA) and Final Exam (Exam).
date - information about the final submission date of the assessment calculated as the number of days since the start of the module-presentation. The starting date of the presentation has number 0 (zero).
weight - weight of the assessment in %. Typically, Exams are treated separately and have the weight 100%; the sum of all other assessments is 100%.
If the information about the final exam date is missing, it is at the end of the last presentation week.

vle.csv
The csv file contains information about the available materials in the VLE. Typically, these are html pages, pdf files, etc. Students have access to these materials online and their interactions with the materials are recorded.
The vle.csv file contains the following columns:

id_site - an identification number of the material.
code_module - an identification code for module.
code_presentation - the identification code of presentation.
activity_type - the role associated with the module material.
week_from - the week from which the material is planned to be used.
week_to - week until which the material is planned to be used.
studentInfo.csv
This file contains demographic information about the students together with their results. File contains the following columns:

code_module - an identification code for a module on which the student is registered.
code_presentation - the identification code of the presentation during which the student is registered on the module.
id_student - a unique identification number for the student.
gender - the student's gender.
region - identifies the geographic region, where the student lived while taking the module-presentation.
highest_education - highest student education level on entry to the module presentation.
imd_band - specifies the Index of Multiple Depravation band of the place where the student lived during the module-presentation.
age_band - band of the student's age.
num_of_prev_attempts - the number times the student has attempted this module.
studied_credits - the total number of credits for the modules the student is currently studying.
disability - indicates whether the student has declared a disability.
final_result - student's final result in the module-presentation.
studentRegistration.csv
This file contains information about the time when the student registered for the module presentation. For students who unregistered the unregistered date is also recorded. File contains five columns:

code_module - an identification code for a module.
code_presentation - the identification code of the presentation.
id_student - a unique identification number for the student.
date_registration - the date of student's registration on the module presentation, this is the number of days measured relative to the start of the module-presentation (e.g. the negative value -30 means that the student registered to module presentation 30 days before it started).
date_unregistration - the student's unregistered date from the module presentation, this is the number of days measured relative to the start of the module-presentation. Students, who completed the course have this field empty. Students who unregistered have Withdrawal as the value of the final_result column in the studentInfo.csv file.
studentAssessment.csv
This file contains the results of students' assessments. If the student does not submit the assessment, no result is recorded. The final exam submissions is missing, if the result of the assessments is not stored in the system.
This file contains the following columns:

id_assessment - the identification number of the assessment.
id_student - a unique identification number for the student.
date_submitted - the date of student submission, measured as the number of days since the start of the module presentation.
is_banked - a status flag indicating that the assessment result has been transferred from a previous presentation.
score - the student's score in this assessment. The range is from 0 to 100. The score lower than 40 is interpreted as Fail. The marks are in the range from 0 to 100.
studentVle.csv
The studentVle.csv file contains information about each student's interactions with the materials in the VLE.
This file contains the following columns:

code_module - an identification code for a module.
code_presentation - the identification code of the module presentation.
id_student - a unique identification number for the student.
id_site - an identification number for the VLE material.
date - the date of student's interaction with the material measured as the number of days since the start of the module-presentation.
sum_click - the number of times a student interacts with the material in that day.

FOR OUR MACHINE LEARNING MODEL

Our Target variable is final result and 
features are
22 Time Series Features Explained
Temporal Feature (1)
week - Time dimension (0-39 weeks), tracks where in the course timeline the student is
​

Student Engagement Features (6)
weekly_clicks - Number of clicks in Virtual Learning Environment (VLE) this specific week
​

cumulative_clicks - Total clicks from week 0 up to current week (shows overall engagement growth)
​

unique_activities - Number of different activity types accessed this week (forums, resources, quizzes, etc.)

weekly_interactions - Count of distinct VLE interactions this week
​

prev_week_clicks - Last week's clicks (lag feature to capture recent behavior change)
​

prev_week_interactions - Last week's interactions (detects sudden drops or spikes)

Rolling Window Features (2)
rolling_avg_clicks_3w - Average clicks over last 3 weeks (smooths out weekly fluctuations to show trends)
​

rolling_avg_interactions_3w - Average interactions over last 3 weeks (identifies sustained engagement patterns)

Assessment Performance Features (3)
cumulative_avg_score - Running average of all assessment scores up to current week
​

cumulative_assessments - Total number of assessments completed so far

cumulative_banked - Number of assessments "banked" (transferred from previous course attempts)

Course Identification Features (2)
code_module_encoded - Which course module (e.g., AAA, BBB, CCC) - encoded as 0, 1, 2, etc.

code_presentation_encoded - Which semester/presentation (e.g., 2013J, 2014B) - different courses have different difficulty

Static Student Demographics (7)
studied_credits - Number of credits student is taking (workload indicator)

date_registration - Days before/after course start when student registered (early = motivated, late = risky)

gender_encoded - Student gender (0 or 1)

region_encoded - Geographic region where student lives

highest_education_encoded - Prior education level (0=No formal quals, 1=Lower than A-level, 2=A-level, 3=HE qualification, 4=Post grad)

imd_band_encoded - Index of Multiple Deprivation (socioeconomic status: 0-10, with 10 being least deprived)

age_band_encoded - Age group (0-25 = code 0, 55+ = code 1, etc.)

disability_encoded - Whether student has declared disability (0 = No, 1 = Yes)

Why These Features Matter for Time Series
Cumulative features (like cumulative_clicks, cumulative_avg_score) show the student's journey - are they improving or declining?
​

Lag features (prev_week_clicks) detect sudden behavior changes - a student who suddenly stops clicking might withdraw
​

Rolling averages smooth out noise - a student might miss one week but their 3-week average shows they're generally engaged
​

Static demographics provide context - older students with families might have different engagement patterns than younger students

We apply a few algorithms and show Top 5 ALgorithms with best accuracy in this

Citation for datasets
Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).
