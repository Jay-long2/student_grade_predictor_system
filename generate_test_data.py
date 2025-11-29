import pandas as pd
import numpy as np
import os

def generate_test_student_data(num_students=30):
    """Generate realistic test student data for bulk predictions"""
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic distributions
    gpa = np.random.normal(2.8, 0.8, num_students)
    gpa = np.clip(gpa, 1.0, 4.0)  # GPA between 1.0 and 4.0
    
    completed_units = np.random.randint(70, 165, num_students)
    
    # Higher GPA students more likely to have internship
    internship_completed = np.random.choice(['Yes', 'No'], num_students, p=[0.6, 0.4])
    
    # Participation based on GPA
    participation_levels = []
    for g in gpa:
        if g > 3.5:
            participation_levels.append('High')
        elif g > 2.5:
            participation_levels.append('Medium')
        else:
            participation_levels.append('Low')
    
    # Discipline and assignment scores correlated with GPA
    discipline_score = np.clip(gpa * 25 + np.random.normal(0, 5, num_students), 0, 100)
    assignment_score = np.clip(gpa * 25 + np.random.normal(0, 5, num_students), 0, 100)
    
    # Create DataFrame
    data = {
        'Student_ID': [f'STU{i+1:03d}' for i in range(num_students)],
        'GPA': [round(g, 1) for g in gpa],
        'Completed_Units': completed_units,
        'Internship_Completed': internship_completed,
        'Participation': participation_levels,
        'Discipline_Score': [round(s) for s in discipline_score],
        'Assignment_Score': [round(s) for s in assignment_score],
    }
    
    df = pd.DataFrame(data)
    return df

# Generate and save the data
if __name__ == "__main__":
    df = generate_test_student_data(30)
    
    # Get the current directory
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'final_test_student_data.csv')
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    
    print("✅ Test CSV file generated successfully!")
    print(f"📁 File saved at: {file_path}")
    print(f"📊 Generated {len(df)} student records")
    
    # Show some statistics
    print(f"\n📈 Data Statistics:")
    print(f"GPA Range: {df['GPA'].min():.1f} - {df['GPA'].max():.1f}")
    print(f"Average GPA: {df['GPA'].mean():.2f}")
    print(f"Internship Completed: {len(df[df['Internship_Completed'] == 'Yes'])} students")
    print(f"Participation - High: {len(df[df['Participation'] == 'High'])}, "
          f"Medium: {len(df[df['Participation'] == 'Medium'])}, "
          f"Low: {len(df[df['Participation'] == 'Low'])}")
    
    print("\n📋 First 5 records:")
    print(df.head())