import pandas as pd
import ast
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import os

# --- Data Loading and Preprocessing ---
# Ensure data files are in the same directory as the script or provide full paths
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory of the current script

try:
    seekers = pd.read_csv(os.path.join(script_dir, "seekers.csv"))
    jobs = pd.read_csv(os.path.join(script_dir, "jobs.csv"))
    distances = pd.read_csv(os.path.join(script_dir, "location_distances.csv"), index_col=0)
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

# --- Load Mw from Part 1 ---
try:
    with open(os.path.join(script_dir, "part1_Mw_value.txt"), "r") as f_mw:
        Mw = float(f_mw.read().strip())
    print(f"Successfully loaded Mw = {Mw} from part1_Mw_value.txt")
except FileNotFoundError:
    print("Error: part1_Mw_value.txt not found. Please run Part 1 first to generate this file.")
    exit()
except ValueError:
    print("Error: Could not parse Mw value from part1_Mw_value.txt. Ensure it contains a valid number.")
    exit()

# --- Pre-calculate Dissimilarity Scores (d_ij) ---
dissimilarity_scores = {}
for i in seekers.index:
    s_quest = seekers.loc[i, "Questionnaire"]
    for j in jobs.index:
        j_quest = jobs.loc[j, "Questionnaire"]
        if len(s_quest) == 20 and len(j_quest) == 20:
            score = sum(abs(s_quest[k] - j_quest[k]) for k in range(20)) / 20.0
            dissimilarity_scores[i, j] = score
        else:
            # Handle cases with unexpected questionnaire length, assign max dissimilarity
            dissimilarity_scores[i, j] = 5.0 
            print(f"Warning: Questionnaire length mismatch for seeker {i}, job {j}. Assigning max dissimilarity.")

# --- ILP Model for Part 2 ---
omega_values = [70, 75, 80, 85, 90, 95, 100]
results_summary = []
all_matches_collections = {}

print("\nStarting Part 2: Minimize Maximum Dissimilarity for various ω values...")

for omega in omega_values:
    model = Model(f"Part2_MinimizeMaxDissimilarity_w{omega}")
    model.setParam('OutputFlag', 0) # Suppress Gurobi output

    # Decision variables
    x = {}
    for i in seekers.index:
        for j in jobs.index:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    M = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=5.0, name="MaxDissimilarity") # M is the max dissimilarity

    # ---------------------------------------------------------------
    # >>> JOB‑TYPE MAPPING + big‑M constant
    # ---------------------------------------------------------------
    job_types = sorted({*seekers["Desired_Job_Type"], *jobs["Job_Type"]})
    job_type_idx = {t: idx for idx, t in enumerate(job_types)}
    M_TYPE = len(job_types) - 1          # maximum possible |difference|

    # ---------------------------------------------------------------
    # >>> Salary + big‑M constant
    # ---------------------------------------------------------------
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

    # PARAMETER
    # m[i, skill] = 1 if seeker i has that skill
    m = {(i, sk): int(sk in seekers.loc[i, "Skills"])
        for i in seekers.index for sk in all_skills}

    # n[j, skill] = 1 if job j requires that skill
    n = {(j, sk): int(sk in jobs.loc[j, "Required_Skills"])
        for j in jobs.index for sk in all_skills}
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Experience
    # ---------------------------------------------------------------
    M_EXP  = 5

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
            exp_diff = s["Experience_Level_Mapped"] - job["Required_Experience_Level_Mapped"]
            model.addConstr(exp_diff + M_EXP * (1 - x[i, j]) >= 0)

            # 5. Location compatibility (big‑M, no aux variable)
            dist_ij = distances.loc[s["Location"], job["Location"]]
            model.addConstr(
                dist_ij - s["Max_Commute_Distance"]
                <= M_DIST * (1 - x[i, j] + job["Is_Remote"])
            )

            # Dissimilarity constraint: d_ij * x_ij <= M
            # This means if x_ij is 1 (match made), then its dissimilarity must be <= M.
            # If x_ij is 0, constraint becomes 0 <= M, which is always true for M >= 0.
            model.addConstr(dissimilarity_scores[i, j] * x[i, j] <= M, name=f"max_dissim_link_{i}_{j}")

    # Constraint: Each seeker can be assigned to at most one job.
    for i in seekers.index:
        model.addConstr(quicksum(x[i, j] for j in jobs.index) <= 1, name=f"seeker_limit_{i}")

    # Constraint: Job capacity.
    for j in jobs.index:
        model.addConstr(quicksum(x[i, j] for i in seekers.index) <= jobs.loc[j, "Num_Positions"], name=f"job_capacity_{j}")

    # Constraint: Total priority weight must be at least ω% of Mw from Part 1.
    min_required_priority = (omega / 100.0) * Mw
    model.addConstr(
        quicksum(jobs.loc[j, "Priority_Weight"] * x[i, j] for i in seekers.index for j in jobs.index) >= min_required_priority,
        name=f"min_priority_target_w{omega}"
    )

    # Objective Function: Minimize the maximum dissimilarity M.
    model.setObjective(M, GRB.MINIMIZE)

    # Solve the model
    model.optimize()

    current_assignments = []
    if model.status == GRB.OPTIMAL:
        achieved_max_dissimilarity = model.objVal
        num_assignments = 0
        current_total_priority = 0
        for i in seekers.index:
            for j in jobs.index:
                if x[i, j].X > 0.5:
                    num_assignments += 1
                    current_total_priority += jobs.loc[j, "Priority_Weight"]
                    current_assignments.append({
                        "Seeker_ID": seekers.loc[i, "Seeker_ID"], 
                        "Job_ID": jobs.loc[j, "Job_ID"],
                        "Seeker_Loc": seekers.loc[i, "Location"],
                        "Job_Loc": jobs.loc[j, "Location"],
                        "Job_Type": jobs.loc[j, "Job_Type"],
                        "Priority": jobs.loc[j, "Priority_Weight"],
                        "Dissimilarity": dissimilarity_scores[i,j]
                    })
        results_summary.append({
            "omega": omega,
            "min_max_dissimilarity": achieved_max_dissimilarity,
            "num_assignments": num_assignments,
            "total_priority": current_total_priority
        })
        all_matches_collections[omega] = current_assignments
        print(f"  ω = {omega}%: Optimal. Max Dissimilarity = {achieved_max_dissimilarity:.4f}, Assignments = {num_assignments}, Total Priority = {current_total_priority:.2f}")
    elif model.status == GRB.INFEASIBLE:
        results_summary.append({"omega": omega, "min_max_dissimilarity": None, "num_assignments": 0, "total_priority": 0})
        all_matches_collections[omega] = []
        print(f"  ω = {omega}%: Infeasible. Could not meet the priority target of {min_required_priority:.2f}.")
    else:
        results_summary.append({"omega": omega, "min_max_dissimilarity": None, "num_assignments": 0, "total_priority": 0})
        all_matches_collections[omega] = []
        print(f"  ω = {omega}%: Optimization stopped with status {model.status}.")

# --- Output and Analysis ---
# 1. Save Summary Table
summary_filename = os.path.join(script_dir, "part2_summary_results.txt")
with open(summary_filename, "w") as f:
    f.write(f"Part 2 Results: Minimizing Maximum Dissimilarity (Mw_Part1 = {Mw:.2f})\n")
    f.write("=" * 90 + "\n")
    f.write(f"{'Omega (%)':<10} {'Min Max Dissim.':<20} {'Num Assignments':<18} {'Total Priority Achieved':<25}\n")
    f.write("-" * 90 + "\n")
    for r in results_summary:
        diss_val = f"{r['min_max_dissimilarity']:.4f}" if r['min_max_dissimilarity'] is not None else "Infeasible"
        f.write(f"{r['omega']:<10} {diss_val:<20} {r['num_assignments']:<18} {r['total_priority']:.2f}\n")
    f.write("=" * 90 + "\n")
print(f"\nSummary results saved to {summary_filename}")

# 2. Graph Findings (ω vs. Min Max Dissimilarity)
feasible_results = [r for r in results_summary if r["min_max_dissimilarity"] is not None]
if feasible_results:
    omegas_plot = [r["omega"] for r in feasible_results]
    dissim_plot = [r["min_max_dissimilarity"] for r in feasible_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(omegas_plot, dissim_plot, marker='o', linestyle='-')
    plt.title("Trade-off: ω vs. Minimum Achievable Maximum Dissimilarity")
    plt.xlabel("ω (% of Part 1 Max Priority Mw)")
    plt.ylabel("Minimum Maximum Dissimilarity (d_ij)")
    plt.grid(True)
    plt.xticks(omega_values) # Ensure all omega values are shown as ticks
    plot_filename = os.path.join(script_dir, "part2_omega_vs_dissimilarity_plot.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # Uncomment to display plot if running in an environment that supports it
else:
    print("No feasible solutions found to generate a plot.")

# 3. We select the ω=90 value for detailed report
chosen_omega = 90
if not any(r["omega"] == chosen_omega and r["min_max_dissimilarity"] is not None for r in feasible_results):
    if feasible_results: # Fallback to the highest omega that was feasible
        chosen_omega = max(r["omega"] for r in feasible_results)
        print(f"\nChosen ω={chosen_omega}% (fallback as initial choice was infeasible or not run). ")
    else:
        chosen_omega = None # No feasible omega
        print("\nNo feasible ω found to select for detailed report.")
else:
    print(f"\nSelected ω = {chosen_omega}% for detailed analysis based on project guidelines or trade-off preference.")

if chosen_omega is None:
    justification = "No feasible omega was found, so no detailed report can be generated."

#print(justification)

# 4. Save detailed assignments for the chosen ω
if chosen_omega is not None and chosen_omega in all_matches_collections:
    detailed_matches = all_matches_collections[chosen_omega]
    detailed_filename = os.path.join(script_dir, f"part2_chosen_omega_{chosen_omega}_detailed_assignments.txt")
    chosen_omega_result = next(r for r in results_summary if r["omega"] == chosen_omega)

    with open(detailed_filename, "w", encoding="utf-8") as f:
        f.write(f"Part 2: Detailed Assignments for Chosen ω = {chosen_omega}%\n")
        f.write(f"Target: Achieve at least {chosen_omega}% of Mw ({Mw:.2f}), which is {(omega/100.0)*Mw:.2f}\n")
        f.write(f"Achieved Max Dissimilarity: {chosen_omega_result['min_max_dissimilarity']:.4f}\n")
        f.write(f"Achieved Total Priority: {chosen_omega_result['total_priority']:.2f}\n")
        f.write(f"Number of Assignments: {chosen_omega_result['num_assignments']}\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Seeker ID':<12} {'Seeker Loc':<11} {'Job ID':<9} {'Job Loc':<8} {'Job Type':<12} {'Priority':<9} {'Dissim.':<10}\n")
        f.write("-" * 100 + "\n")
        
        if detailed_matches:
            for match in detailed_matches:
                f.write(f"{match['Seeker_ID']:<12} {match['Seeker_Loc']:<11} {match['Job_ID']:<9} {match['Job_Loc']:<8} "
                        f"{match['Job_Type']:<12} {match['Priority']:<9.2f} {match['Dissimilarity']:<10.4f}\n")
        else:
            f.write("No assignments made for this omega value.\n")
        f.write("=" * 100 + "\n")
        #f.write(justification + "\n")
    print(f"Detailed assignments for ω = {chosen_omega}% saved to {detailed_filename}")
else:
    print(f"Could not generate detailed report for chosen_omega={chosen_omega} as it was not feasible or not found.")

print("\nPart 2 processing complete.")
