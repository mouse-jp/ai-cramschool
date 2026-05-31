import json

db_path = "past_exams_db.json"
with open(db_path, "r", encoding="utf-8") as f:
    db = json.load(f)

for course_key, course_data in db.items():
    for uni_key, uni_data in course_data.items():
        for fac_key, fac_data in uni_data.items():
            for year_key, year_data in fac_data.items():
                for method_key, method_data in year_data.items():
                    # 熟語データの中にある 'quotes' をすべて消去
                    if "idioms" in method_data:
                        for idiom_key, idiom_val in method_data["idioms"].items():
                            idiom_val.pop("quotes", None)

with open(db_path, "w", encoding="utf-8") as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print("✨ データベースの著作権クリーンアップが完了しました！")