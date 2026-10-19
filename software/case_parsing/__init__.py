"""FAERS case parsing: drug-record deduplication and structured narrative -> DataFrames."""
import re

import pandas as pd
from rich.console import Console

console = Console()


# ════════════════════════════════════════════════════════════════════════════
# Deduplication
# ════════════════════════════════════════════════════════════════════════════
import re


def deduplicate_drugs(text):
    """Keep one record per drug (the one with the most non-missing fields) and
    return the case as one "Field: value" line per field, ready for `process_case`."""
    if not text.strip():
        return "The input box is empty."

    # 1. Define the full list of keys to find in your text
    all_keys = [
        'Case ID', 'Case Version', 'FDA Code', 'Event Date', 'Manufacture Date',
        'Initial FDA Date', 'FDA Date', 'Report Code', 'Authorization Number',
        'Manufacturer Number', 'Manufacturer Sender', 'Literature Reference',
        'Age', 'Age Code', 'Age Group', 'Sex', 'E Sub', 'Weight', 'Weight Code',
        'Report Date', 'To Manufacturer', 'Occupation Code', 'Reporter Country',
        'Occurrence Country', 'Adverse Reaction(s)', 'Case ID Number',
        'Drug Sequence', 'Role Code', 'Drug Name', 'Product AI', 'Val VBM',
        'Route', 'Dose VBM', 'Cumulative Dose (Char)', 'Cumulative Dose (Unit)',
        'Dechal', 'Rechal', 'Lot Number', 'Expiration Date', 'NDA Number',
        'Dose Amount', 'Dose Unit', 'Dose Form', 'Dose Frequency',
        'Indication for Use', 'Therapy Start Date', 'Therapy End Date',
        'Duration', 'Duration Code'
    ]

    # 2. Extract values using Key Anchors
    pattern = '|'.join([re.escape(k) for k in all_keys])
    matches = list(re.finditer(f'({pattern}):', text))

    data_dict = {}
    for i in range(len(matches)):
        start_idx = matches[i].end()
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(text)
        key = matches[i].group(1)
        value = text[start_idx:end_idx].strip()
        data_dict[key] = value

    if 'Drug Name' not in data_dict:
        return "Error: Could not find 'Drug Name'. Ensure the input has 'Drug Name:'"

    # 3. List of fields that are "aligned" (semicolon separated)
    aligned_fields = [
        'Drug Sequence', 'Role Code', 'Drug Name', 'Product AI', 'Val VBM',
        'Route', 'Dose VBM', 'Cumulative Dose (Char)', 'Cumulative Dose (Unit)',
        'Dechal', 'Rechal', 'Lot Number', 'Expiration Date', 'NDA Number',
        'Dose Amount', 'Dose Unit', 'Dose Form', 'Dose Frequency',
        'Indication for Use', 'Therapy Start Date', 'Therapy End Date',
        'Duration', 'Duration Code'
    ]

    # 4. Group by Drug Name and pick the best index
    drug_names = [d.strip() for d in data_dict['Drug Name'].split(';')]
    best_indices = {} # {drug_name: index}

    for i, name in enumerate(drug_names):
        # Calculate info score for this specific index
        current_score = 0
        for field in aligned_fields:
            vals = [v.strip() for v in data_dict.get(field, "").split(';')]
            if i < len(vals) and vals[i].upper() != "NA" and vals[i] != "":
                current_score += 1

        # If we haven't seen this drug, or this version is better, save index
        if name not in best_indices:
            best_indices[name] = (i, current_score)
        else:
            if current_score > best_indices[name][1]:
                best_indices[name] = (i, current_score)

    # Sorted list of the original indices we want to keep
    target_indices = sorted([val[0] for val in best_indices.values()])

    # 5. Reconstruct the text structure
    output_parts = []
    for key in all_keys:
        if key in data_dict:
            if key in aligned_fields:
                # Filter values to only keep the "best" indices
                raw_vals = [v.strip() for v in data_dict[key].split(';')]
                # Ensure we have at least NA if index is missing
                filtered_vals = []
                for idx in target_indices:
                    if idx < len(raw_vals):
                        v = raw_vals[idx]
                        filtered_vals.append(v if v != "" else "NA")
                    else:
                        filtered_vals.append("NA")

                new_value = " ; ".join(filtered_vals)
                output_parts.append(f"{key}: {new_value}")
            else:
                # Static fields (Age, Case ID, etc.) stay as they are
                output_parts.append(f"{key}: {data_dict[key]}")

    return " \n".join(output_parts)


# Name used in the original notebook
process_case_text_to_text = deduplicate_drugs


# ════════════════════════════════════════════════════════════════════════════
# Parser
# ════════════════════════════════════════════════════════════════════════════


def parse_case_text(case_text):
    lines = case_text.strip().split("\n")
    data = {}

    for line in lines:
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()

    case_info = {
        "Case ID": data.get("Case ID", "NA"),
        "Case Version": data.get("Case Version", "NA"),
        "FDA Code": data.get("FDA Code", "NA"),
        "Event Date": data.get("Event Date", "NA"),
        "Manufacture Date": data.get("Manufacture Date", "NA"),
        "Initial FDA Date": data.get("Initial FDA Date", "NA"),
        "FDA Date": data.get("FDA Date", "NA"),
        "Report Code": data.get("Report Code", "NA"),
        "Authorization Number": data.get("Authorization Number", "NA"),
        "Manufacturer Number": data.get("Manufacturer Number", "NA"),
        "Manufacturer Sender": data.get("Manufacturer Sender", "NA"),
        "Literature Reference": data.get("Literature Reference", "NA"),
        "Age": data.get("Age", "NA"),
        "Age Code": data.get("Age Code", "NA"),
        "Age Group": data.get("Age Group", "NA"),
        "Sex": data.get("Sex", "NA"),
        "E Sub": data.get("E Sub", "NA"),
        "Weight": data.get("Weight", "NA"),
        "Weight Code": data.get("Weight Code", "NA"),
        "Report Date": data.get("Report Date", "NA"),
        "To Manufacturer": data.get("To Manufacturer", "NA"),
        "Occupation Code": data.get("Occupation Code", "NA"),
        "Reporter Country": data.get("Reporter Country", "NA"),
        "Occurrence Country": data.get("Occurrence Country", "NA"),
        "Case ID Number": data.get("Case ID Number", "NA"),
        "EDSS": data.get("EDSS", "NA"),
    }

    adverse_reactions_raw = data.get("Adverse Reaction(s)", "")
    adverse_reactions = [r.strip() for r in adverse_reactions_raw.split(";") if r.strip()]

    drug_fields = [
        "Drug Sequence", "Role Code", "Drug Name", "Product AI", "Val VBM",
        "Route", "Dose VBM", "Cumulative Dose (Char)", "Cumulative Dose (Unit)",
        "Dechal", "Rechal", "Lot Number", "Expiration Date", "NDA Number",
        "Dose Amount", "Dose Unit", "Dose Form", "Dose Frequency",
        "Indication for Use", "Therapy Start Date", "Therapy End Date",
        "Duration", "Duration Code"
    ]

    drug_data = {}

    if "Drug Sequence" in data:
        sequences = [v.strip() for v in data["Drug Sequence"].split(";")]
        max_drugs = len(sequences)
        drug_data["Drug Sequence"] = sequences
    else:
        max_drugs = 0

    for field in drug_fields:
        if field == "Drug Sequence":
            continue

        if field in data:
            values = [v.strip() for v in data[field].split(";")]

            if field == "Indication for Use" and len(values) > max_drugs:
                values = values[:max_drugs]

            while len(values) < max_drugs:
                values.append("NA")

            drug_data[field] = values[:max_drugs]

    drugs_list = []
    for i in range(max_drugs):
        drug = {}
        for field in drug_fields:
            if field in drug_data and i < len(drug_data[field]):
                drug[field] = drug_data[field][i]
            else:
                drug[field] = "NA"
        drugs_list.append(drug)

    return case_info, adverse_reactions, drugs_list


def create_dataframes_from_case(case_info, adverse_reactions, drugs_list):
    df_case = pd.DataFrame([case_info])

    if adverse_reactions:
        df_reactions = pd.DataFrame({
            "Case ID": [case_info["Case ID"]] * len(adverse_reactions),
            "Adverse Reaction": adverse_reactions
        })
    else:
        df_reactions = pd.DataFrame(columns=["Case ID", "Adverse Reaction"])

    if drugs_list:
        df_drugs = pd.DataFrame(drugs_list)
        df_drugs.insert(0, "Case ID", case_info["Case ID"])
    else:
        df_drugs = pd.DataFrame()

    if not df_drugs.empty:
        df_comprehensive = df_drugs.copy()
        for key, value in case_info.items():
            if key not in df_comprehensive.columns:
                df_comprehensive[key] = value
    else:
        df_comprehensive = pd.DataFrame()

    return df_case, df_reactions, df_drugs, df_comprehensive


def process_case(case_text, display=True):
    case_info, adverse_reactions, drugs_list = parse_case_text(case_text)
    df_case, df_reactions, df_drugs, df_comprehensive = create_dataframes_from_case(
        case_info, adverse_reactions, drugs_list
    )

    if display:
        console.print("\n" + "=" * 80, style="bold cyan")
        console.print("FDA CASE PARSED INTO DATAFRAMES", style="bold cyan")
        console.print("=" * 80, style="bold cyan")
        console.print(f"\n✅ Case ID: {case_info['Case ID']}", style="green")
        console.print(f"✅ Drugs: {len(drugs_list)}", style="green")
        console.print(f"✅ Adverse Reactions: {len(adverse_reactions)}", style="green")

    return df_case, df_reactions, df_drugs, df_comprehensive
