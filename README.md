# Job Matching Optimization with Gurobi

This project implements a two-part Integer Linear Programming (ILP) solution to optimize job assignments between seekers and job openings. Developed as part of the IE400 Principles of Engineering Management course, it focuses on maximizing assignment quality using Gurobi Optimizer.

## 📊 Project Overview

The goal is to assign job seekers to jobs in an optimal way, balancing two competing objectives:

1. **Part 1:** Maximize the total priority-weighted matches.
2. **Part 2:** Minimize the worst (maximum) dissimilarity between assigned seeker-job pairs, while still achieving at least ω% of the total priority weight from Part 1.

## 🧠 Key Concepts

- **Priority Weight (Wj):** Indicates the importance of filling a job.
- **Dissimilarity Score (dij):** Average absolute difference between seeker and job 20-question survey responses.
- **Big-M Constraints:** Used to enforce logical conditions like skill, location, and experience compatibility.

## 📁 Data Files

- `seekers.csv`: Contains job seeker information.
- `jobs.csv`: Contains job opening details.
- `location_distances.csv`: Distance matrix between locations (A–F).

## 🧮 Part 1: Maximize Priority-Weighted Matches

**Objective:**  
Maximize the total sum of filled jobs’ priority weights.

**Constraints:**

- Each seeker can be assigned to **at most one** job.
- Jobs cannot exceed their available number of positions.
- Assignments must satisfy:
  - Job type compatibility
  - Salary expectations
  - Required skills
  - Experience level
  - Commute distance or remote job status

### Output
- Optimal total priority weight `Mw` (e.g., 240.00)
- List of assignments saved to `part1_results_detailed.txt`

## 🧮 Part 2: Minimize Maximum Dissimilarity

**Objective:**  
Minimize the **maximum dissimilarity score** among all matches, subject to a minimum total priority requirement.

**Additional Constraint:**
- The total priority must be at least ω% of `Mw`.

**ω Values Tested:**  
{70, 75, 80, 85, 90, 95, 100}

### Output
- Summary in `part2_summary_results.txt`
- Trade-off plot: `part2_omega_vs_dissimilarity_plot.png`
- Detailed results for selected ω = 90% in `part2_chosen_omega_90_detailed_assignments.txt`

## 📈 Results Summary

| ω (%) | Max Dissimilarity | Assignments | Total Priority |
|-------|--------------------|-------------|----------------|
| 70    | 2.0000             | 31          | 168.00         |
| 75    | 2.0000             | 33          | 180.00         |
| 80    | 2.1000             | 36          | 195.00         |
| 85    | 2.1500             | 41          | 221.00         |
| 90    | 2.1500             | 41          | 221.00         |
| 95    | 2.3000             | 43          | 229.00         |
| 100   | 2.5000             | 46          | 240.00         |

**Chosen ω = 90%** represents a well-balanced solution with high priority and low dissimilarity.

## ⚙️ How to Run

1. Install Gurobi and obtain an academic license.
2. Ensure the CSV files are in the same directory as the Python scripts.
3. Run `part1.py` to generate `Mw`.
4. Run `part2.py` to evaluate various ω values.

## 👥 Group Members

- Emre Furkan Akyol
- İbrahim Çaycı
- Batu Uzun

## 📄 License

This project was created for academic purposes as part of the IE400 course at Bilkent University.

---

