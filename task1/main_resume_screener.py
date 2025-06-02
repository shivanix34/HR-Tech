import os
import csv
import re
from script import PROMPT_TEMPLATE
from api import call_azure_openai
from pdf import extract_text_from_pdf
import glob

def parse_response(result):
    name = match_score = conclusion = skills = "Not Extracted"
    summary_lines = []

    lines = result.splitlines()
    for line in lines:
        line = line.strip()

        clean_line = re.sub(r'^[•\-●▪️\*]+\s*', '', line)

        if "Candidate Name" in line:
            name = line.split(":", 1)[-1].strip()
        elif "Match Score" in line:
            score_str = line.split(":", 1)[-1].strip()
            try:
                match_score = int(re.findall(r'\d+', score_str)[0])
            except:
                match_score = 0
        elif "Top 5 Matched Skills" in line:
            skills = line.split(":", 1)[-1].strip()
        elif clean_line and re.match(r'^[A-Za-z0-9 ,./]+$', clean_line):
            summary_lines.append(clean_line)
        elif "Conclusion" in line:
            conclusion_raw = line.split(":", 1)[-1].strip()
            if "Yes" in conclusion_raw:
                conclusion = "Yes. Candidate meets the requirements"
            elif "No" in conclusion_raw:
                conclusion = "No. Candidate lacks core criteria"
            else:
                conclusion = "Unclear. Review needed"

    return name, match_score, skills, summary_lines[:5], conclusion

def main():
    jd_path = "data/software_engineer_jd.txt"
    resume_paths = glob.glob("data/resume*.pdf")

    with open(jd_path, "r", encoding="utf-8") as f:
        job_description = f.read()

    output_rows = [
        ["Resume", "Candidate Name", "Match Score", "Top 5 Skills", "Summary (5 Points)", "Conclusion"]
    ]

    for resume_file in resume_paths:
        print(f"Processing: {resume_file}")
        resume_text = extract_text_from_pdf(resume_file)

        prompt = PROMPT_TEMPLATE.format(
            job_description=job_description,
            resume_text=resume_text
        )

        try:
            result = call_azure_openai(prompt)
            print(result)

            name, match_score, skills, summary, conclusion = parse_response(result)
            summary_formatted = "\n".join([f"{i+1}. {line}" for i, line in enumerate(summary)])

            output_rows.append([
                os.path.basename(resume_file),
                name,
                match_score,
                skills,
                summary_formatted,
                conclusion
            ])

        except Exception as e:
            print(f"Error processing {resume_file}: {e}")
            output_rows.append([
                os.path.basename(resume_file), "Error", 0, "Error", "Error", str(e)
            ])

    header = output_rows[0]
    data_rows = output_rows[1:]
    sorted_data = sorted(data_rows, key=lambda x: int(x[2]), reverse=True)
    final_rows = [header] + sorted_data

    with open("resume_match_results.csv", "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(final_rows)

    print("\n Matching complete. Output saved to: resume_match_results.csv")

if __name__ == "__main__":
    main()
