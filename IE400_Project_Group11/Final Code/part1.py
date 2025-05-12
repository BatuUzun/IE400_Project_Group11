\
import pandas as pd
import ast
from gurobipy import Model, GRB, quicksum

# --- Data Loading and Preprocessing ---
try:
    seekers = pd.read_csv("seekers.csv")
    jobs = pd.read_csv("jobs.csv")
    distances = pd.read_csv("location_distances.csv", index_col=0)
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please ensure seekers.csv, jobs.csv, and location_distances.csv are in the same directory as this script.")
    exit()

# Parse string representations of lists
for df, cols in [(seekers, ["Skills", "Questionnaire"]), (jobs, ["Required_Skills", "Questionnaire"])]:
    for col in cols:
        try:
            df[col] = df[col].apply(ast.literal_eval)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing column {col} in {df.name if hasattr(df, 'name') else 'DataFrame'}: {e}")
            exit()

# Experience level mapping
exp_map = {"Entry-level": 1, "Mid-level": 2, "Senior": 3, "Lead": 4, "Manager": 5}
seekers["Experience_Level_Mapped"] = seekers["Experience_Level"].map(exp_map)
jobs["Required_Experience_Level_Mapped"] = jobs["Required_Experience_Level"].map(exp_map)

# --- ILP Model for Part 1 ---
model = Model("Part1_MaximizePriorityWeightedMatches")
model.setParam('OutputFlag', 0) # Suppress Gurobi output

# Decision variables: x_ij = 1 if seeker i is assigned to job j, 0 otherwise
x = {}
for i in seekers.index:
    for j in jobs.index:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")


# ---------------------------------------------------------------
# >>> JOB‑TYPE MAPPING + big‑M constant
# ---------------------------------------------------------------
job_types = sorted({*seekers["Desired_Job_Type"], *jobs["Job_Type"]})
job_type_idx = {t: idx for idx, t in enumerate(job_types)}
M_TYPE = len(job_types) - 1          # maximum possible |difference|

# Big‑M constants
M_SALARY = max(seekers["Min_Desired_Salary"].max(),
               jobs["Salary_Range_Max"].max())

# ---------------------------------------------------------------
# >>> NEW PARAMETER SET‑UP FOR SKILL VECTOR TESTS <<<
# ---------------------------------------------------------------
# Full catalogue of skills appearing anywhere
all_skills = sorted({
    skill
    for lst in seekers["Skills"].tolist() + jobs["Required_Skills"].tolist()
    for skill in lst
})

# m[i, skill] = 1 if seeker i has that skill
m = {(i, sk): int(sk in seekers.loc[i, "Skills"])
     for i in seekers.index for sk in all_skills}

# n[j, skill] = 1 if job j requires that skill
n = {(j, sk): int(sk in jobs.loc[j, "Required_Skills"])
     for j in jobs.index for sk in all_skills}
# ---------------------------------------------------------------

# ------------------------------------------------------------------
# >>> Big‑M for distance
# ------------------------------------------------------------------
# If the distance matrix is complete, the safest Big‑M is simply
#   the maximum entry in that matrix.
# Add a small buffer (e.g. +1) just to be 100 % safe.
if distances.size:                          # matrix not empty
    M_DIST = distances.values.max() + 1
else:                                       # should never happen
    M_DIST = 0
# ------------------------------------------------------------------

M_DIST   = distances.values.max()

for i in seekers.index:
    s = seekers.loc[i]
    for j in jobs.index:
        job = jobs.loc[j]

        # 1. Job‑type compatibility (big‑M, no aux variable)
        seeker_type_code = job_type_idx[s["Desired_Job_Type"]]
        job_type_code    = job_type_idx[job["Job_Type"]]

        # Enforce equality when x_ij = 1
        model.addConstr(seeker_type_code - job_type_code
                        <= M_TYPE * (1 - x[i, j]))
        model.addConstr(job_type_code - seeker_type_code
                        <= M_TYPE * (1 - x[i, j]))


        # 2. Salary compatibility (big‑M)
        model.addConstr(s["Min_Desired_Salary"] - job["Salary_Range_Min"]
                        + M_SALARY * (1 - x[i, j]) >= 0)
        model.addConstr(job["Salary_Range_Max"] - s["Min_Desired_Salary"]
                        + M_SALARY * (1 - x[i, j]) >= 0)
        
        # 3. Skills compatibility (vector style)
        # Enforce: for every required skill  n_j,sk = 1  ⇒  x_ij ≤ m_i,sk
        for sk in all_skills:
            model.addConstr(x[i, j] <= m[i, sk] + (1 - n[j, sk]))

        # 4. Experience Compatibility
        M_EXP  = 5
        exp_diff = s["Experience_Level_Mapped"] - job["Required_Experience_Level_Mapped"]
        model.addConstr(exp_diff + M_EXP * (1 - x[i, j]) >= 0)

        # 5. Location compatibility (big‑M, no aux variable)
        dist_ij = distances.loc[s["Location"], job["Location"]]
        model.addConstr(
            dist_ij - s["Max_Commute_Distance"]
            <= M_DIST * (1 - x[i, j] + job["Is_Remote"])
        )


# Constraints
# Each job seeker can be assigned to at most one job opening.
for i in seekers.index:
    model.addConstr(quicksum(x[i, j] for j in jobs.index) <= 1, name=f"seeker_limit_{i}")

# The number of seekers assigned to a job opening cannot exceed its available positions (Pj).
for j in jobs.index:
    model.addConstr(quicksum(x[i, j] for i in seekers.index) <= jobs.loc[j, "Num_Positions"], name=f"job_capacity_{j}")

# Objective Function: Maximize the total sum of priority weights (wj) of the filled positions.
model.setObjective(
    quicksum(jobs.loc[j, "Priority_Weight"] * x[i, j] for i in seekers.index for j in jobs.index),
    GRB.MAXIMIZE
)

# Solve the model
model.optimize()

# --- Results ---
if model.status == GRB.OPTIMAL:
    Mw = model.objVal
    print(f"Part 1: Optimal solution found.")
    print(f"Maximum Total Priority Weight (Mw): {Mw:.2f}")

    assignments = []
    for i in seekers.index:
        for j in jobs.index:
            if x[i, j].X > 0.5: # If x_ij is 1
                assignments.append((seekers.loc[i, "Seeker_ID"], jobs.loc[j, "Job_ID"]))
    
    print(f"Number of assignments made: {len(assignments)}")

    # Write detailed log to file
    output_filename = 'part1_results_detailed.txt'
    with open(output_filename, 'w') as f:
        f.write("Part 1 Results: Maximize Priority-Weighted Matches\n")
        f.write("=" * 70 + "\n")
        f.write(f"Maximum Total Priority Weight (Mw): {Mw:.2f}\n")
        f.write(f"Number of assignments: {len(assignments)}\n\n")
        f.write("Detailed Assignments (Seeker_ID -> Job_ID):\n")
        f.write(f"{'Seeker ID':<12} {'Seeker Loc':<11} {'Job ID':<9} {'Job Loc':<8} {'Job Type':<12} {'Priority':<9}\n")
        f.write("-" * 70 + "\n")
        
        total_priority_check = 0
        for s_idx, j_idx in [(s_idx, j_idx) for s_idx in seekers.index for j_idx in jobs.index if x[s_idx, j_idx].X > 0.5]:
            seeker_info = seekers.loc[s_idx]
            job_info = jobs.loc[j_idx]
            f.write(f"{seeker_info['Seeker_ID']:<12} {seeker_info['Location']:<11} {job_info['Job_ID']:<9} {job_info['Location']:<8} {job_info['Job_Type']:<12} {job_info['Priority_Weight']:<9.2f}\n")
            total_priority_check += job_info['Priority_Weight']
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Sum of priorities in assignments (check): {total_priority_check:.2f}\n")
    print(f"Detailed results saved to {output_filename}")

elif model.status == GRB.INFEASIBLE:
    print("Part 1: Model is infeasible. No solution found.")
    # Compute IIS to find out why
    # model.computeIIS()
    # model.write("part1_model_iis.ilp")
    # print("IIS written to part1_model_iis.ilp")
else:
    print(f"Part 1: Optimization was stopped with status {model.status}")

# Store Mw for Part 2 (e.g., by writing to a file or returning)
# For this script, we'll just print it. Part 2 will need this value.
if 'Mw' in locals():
    with open("part1_Mw_value.txt", "w") as f_mw:
        f_mw.write(str(Mw))
    print(f"Mw value ({Mw:.2f}) saved to part1_Mw_value.txt for Part 2.")
