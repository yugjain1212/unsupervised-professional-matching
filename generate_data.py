"""Generate synthetic professional profiles for the ML project."""

import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)
N = 50000

FIRST_NAMES = [
    "James",
    "Mary",
    "John",
    "Patricia",
    "Robert",
    "Jennifer",
    "Michael",
    "Linda",
    "William",
    "Elizabeth",
    "David",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Christopher",
    "Karen",
    "Charles",
    "Lisa",
    "Daniel",
    "Nancy",
    "Matthew",
    "Betty",
    "Anthony",
    "Margaret",
    "Mark",
    "Sandra",
    "Donald",
    "Ashley",
    "Steven",
    "Kimberly",
    "Paul",
    "Emily",
    "Andrew",
    "Donna",
    "Joshua",
    "Michelle",
]
LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Perez",
    "Thompson",
    "White",
    "Harris",
    "Sanchez",
    "Clark",
    "Ramirez",
    "Lewis",
    "Robinson",
    "Walker",
    "Young",
    "Allen",
    "King",
    "Wright",
    "Scott",
    "Torres",
    "Nguyen",
    "Hill",
    "Flores",
]

ROLES = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "DevOps Engineer",
    "ML Engineer",
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "Data Analyst",
    "QA Engineer",
    "UI/UX Designer",
    "Systems Architect",
    "Security Engineer",
    "Cloud Engineer",
    "Data Engineer",
    "Scrum Master",
    "Technical Lead",
    "Engineering Manager",
    "Solutions Architect",
    "Platform Engineer",
    "Site Reliability Engineer",
    "Research Scientist",
    "Business Analyst",
    "CTO",
]

INDUSTRIES = [
    "Technology",
    "Finance",
    "Healthcare",
    "Education",
    "E-commerce",
    "Consulting",
    "Energy",
    "Media",
    "Telecommunications",
    "Manufacturing",
    "Automotive",
    "Aerospace",
    "Retail",
    "Insurance",
    "Biotechnology",
]

SKILLS_POOL = {
    "Technology": [
        "python",
        "java",
        "javascript",
        "react",
        "aws",
        "docker",
        "kubernetes",
        "git",
        "sql",
        "linux",
        "api",
        "microservices",
        "cicd",
        "tensorflow",
        "pytorch",
    ],
    "Finance": [
        "python",
        "sql",
        "excel",
        "risk modeling",
        "quantitative analysis",
        "r",
        "sas",
        "tableau",
        "power bi",
        "bloomberg",
        "financial modeling",
        "vba",
        "matlab",
        "machine learning",
        "statistics",
    ],
    "Healthcare": [
        "python",
        "r",
        "clinical trials",
        "epidemiology",
        "biostatistics",
        "sas",
        "sql",
        "machine learning",
        "data visualization",
        "healthcare analytics",
        "hl7",
        "fhir",
        "tableau",
        "statistics",
        "spss",
    ],
    "Education": [
        "python",
        "javascript",
        "react",
        "instructional design",
        "lms",
        "moodle",
        "canvas",
        "e-learning",
        "curriculum development",
        "assessment",
        "sql",
        "html",
        "css",
        "node.js",
        "mongodb",
    ],
    "E-commerce": [
        "python",
        "javascript",
        "react",
        "node.js",
        "aws",
        "shopify",
        "seo",
        "analytics",
        "ab testing",
        "sql",
        "marketing automation",
        "crm",
        "html",
        "css",
        "mongodb",
    ],
    "Consulting": [
        "python",
        "sql",
        "tableau",
        "power bi",
        "excel",
        "project management",
        "agile",
        "stakeholder management",
        "business analysis",
        "strategy",
        "r",
        "statistics",
        "machine learning",
        "data visualization",
        "aws",
    ],
    "Energy": [
        "python",
        "matlab",
        "r",
        "data analysis",
        "sql",
        "machine learning",
        "optimization",
        "simulation",
        "power systems",
        "renewable energy",
        "scada",
        "plc",
        "tableau",
        "statistics",
        "engineering",
    ],
    "Media": [
        "python",
        "javascript",
        "react",
        "video editing",
        "adobe creative suite",
        "seo",
        "content strategy",
        "analytics",
        "html",
        "css",
        "node.js",
        "aws",
        "sql",
        "marketing",
        "photography",
    ],
    "Telecommunications": [
        "python",
        "java",
        "sql",
        "networking",
        "5g",
        "cloud",
        "aws",
        "azure",
        "linux",
        "docker",
        "kubernetes",
        "cicd",
        "api",
        "microservices",
        "security",
    ],
    "Manufacturing": [
        "python",
        "sql",
        "matlab",
        "cad",
        "lean manufacturing",
        "six sigma",
        "plc",
        "scada",
        "automation",
        "robotics",
        "iot",
        "data analysis",
        "statistics",
        "machine learning",
        "tableau",
    ],
    "Automotive": [
        "python",
        "c++",
        "matlab",
        "cad",
        "embedded systems",
        "autonomous driving",
        "machine learning",
        "computer vision",
        "ros",
        "sensor fusion",
        "linux",
        "sql",
        "data analysis",
        "statistics",
        "simulation",
    ],
    "Aerospace": [
        "python",
        "c++",
        "matlab",
        "cad",
        "systems engineering",
        "simulation",
        "aerodynamics",
        "control systems",
        "signal processing",
        "data analysis",
        "linux",
        "sql",
        "machine learning",
        "statistics",
        "engineering",
    ],
    "Retail": [
        "python",
        "sql",
        "excel",
        "tableau",
        "power bi",
        "analytics",
        "ab testing",
        "customer segmentation",
        "forecasting",
        "inventory management",
        "r",
        "statistics",
        "machine learning",
        "data visualization",
        "aws",
    ],
    "Insurance": [
        "python",
        "sql",
        "excel",
        "actuarial science",
        "risk assessment",
        "r",
        "sas",
        "tableau",
        "power bi",
        "statistics",
        "machine learning",
        "data analysis",
        "financial modeling",
        "vba",
        "claims analysis",
    ],
    "Biotechnology": [
        "python",
        "r",
        "matlab",
        "bioinformatics",
        "genomics",
        "statistics",
        "machine learning",
        "sql",
        "data analysis",
        "laboratory techniques",
        "spss",
        "tableau",
        "sas",
        "biostatistics",
        "clinical trials",
    ],
}

SENIORITY_LEVELS = ["entry", "mid", "senior", "executive"]
REMOTE_PREFS = ["onsite", "hybrid", "remote"]
US_STATES = [
    "CA",
    "NY",
    "TX",
    "FL",
    "IL",
    "PA",
    "OH",
    "GA",
    "NC",
    "MI",
    "NJ",
    "VA",
    "WA",
    "AZ",
    "MA",
    "TN",
    "IN",
    "MO",
    "MD",
    "WI",
    "CO",
    "MN",
    "SC",
    "AL",
    "LA",
    "KY",
    "OR",
    "OK",
    "CT",
    "UT",
]
CITIES = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
    "Austin",
    "Jacksonville",
    "Fort Worth",
    "Columbus",
    "Charlotte",
    "San Francisco",
    "Indianapolis",
    "Seattle",
    "Denver",
    "Washington",
]
COUNTRIES = [
    "US",
    "UK",
    "Canada",
    "India",
    "Australia",
    "Germany",
    "France",
    "Singapore",
]

GOALS_POOL = [
    "career growth",
    "networking",
    "mentorship",
    "skill development",
    "leadership",
    "industry insights",
    "job opportunities",
    "collaboration",
    "knowledge sharing",
    "startup advice",
    "technical expertise",
    "research partnership",
]
NEEDS_POOL = [
    "technical guidance",
    "career advice",
    "industry connections",
    "skill mentoring",
    "project collaboration",
    "research opportunities",
    "job referrals",
    "startup mentorship",
    "leadership coaching",
    "peer feedback",
]
OFFER_POOL = [
    "technical mentoring",
    "industry experience",
    "network connections",
    "code review",
    "career guidance",
    "project collaboration",
    "research expertise",
    "startup advice",
    "leadership coaching",
    "skill training",
]

COMPANIES = [
    "Google",
    "Microsoft",
    "Amazon",
    "Apple",
    "Meta",
    "Netflix",
    "Tesla",
    "IBM",
    "Oracle",
    "Salesforce",
    "Adobe",
    "Intel",
    "Cisco",
    "SAP",
    "Accenture",
    "Deloitte",
    "McKinsey",
    "Goldman Sachs",
    "JP Morgan",
    "Wells Fargo",
]


def gen_experience(seniority, industry):
    n_roles = {
        "entry": np.random.choice([1, 2], p=[0.7, 0.3]),
        "mid": np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2]),
        "senior": np.random.choice([3, 4, 5, 6], p=[0.2, 0.3, 0.3, 0.2]),
        "executive": np.random.choice([5, 6, 7, 8], p=[0.2, 0.3, 0.3, 0.2]),
    }[seniority]
    roles = []
    for i in range(n_roles):
        dur_val = np.random.choice(
            [6, 12, 18, 24, 36, 48], p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05]
        )
        unit = "months" if dur_val <= 24 else "years" if dur_val >= 36 else "months"
        if unit == "years":
            dur_val = dur_val // 12
        roles.append(
            {
                "title": np.random.choice(ROLES),
                "company": np.random.choice(COMPANIES),
                "duration": f"{dur_val} {unit}",
            }
        )
    return roles


def gen_education(seniority):
    if seniority == "entry":
        deg = np.random.choice(["Bachelor", "Master"], p=[0.7, 0.3])
    elif seniority == "mid":
        deg = np.random.choice(["Bachelor", "Master", "PhD"], p=[0.3, 0.5, 0.2])
    elif seniority == "senior":
        deg = np.random.choice(["Bachelor", "Master", "PhD"], p=[0.2, 0.5, 0.3])
    else:
        deg = np.random.choice(
            ["Bachelor", "Master", "PhD", "MBA"], p=[0.1, 0.4, 0.3, 0.2]
        )
    return [
        {
            "degree": deg,
            "field": np.random.choice(
                [
                    "Computer Science",
                    "Engineering",
                    "Mathematics",
                    "Business",
                    "Data Science",
                    "Information Technology",
                ]
            ),
            "year": int(np.random.uniform(2010, 2024)),
        }
    ]


rows = []
for i in range(N):
    industry = np.random.choice(INDUSTRIES)
    seniority = np.random.choice(SENIORITY_LEVELS, p=[0.3, 0.35, 0.25, 0.1])
    role = np.random.choice(ROLES)

    # Skills: 60% from industry pool, 40% random
    n_skills = np.random.randint(3, 12)
    industry_skills = SKILLS_POOL[industry]
    other_skills = [
        s for pool in SKILLS_POOL.values() for s in pool if s not in industry_skills
    ]
    n_industry = max(1, int(n_skills * 0.6))
    n_other = n_skills - n_industry
    skills = list(
        np.random.choice(
            industry_skills, size=min(n_industry, len(industry_skills)), replace=False
        )
    )
    if n_other > 0 and other_skills:
        skills.extend(
            list(
                np.random.choice(
                    other_skills, size=min(n_other, len(other_skills)), replace=False
                )
            )
        )
    np.random.shuffle(skills)

    # Years experience correlated with seniority
    exp_map = {"entry": (0, 3), "mid": (3, 7), "senior": (7, 15), "executive": (15, 30)}
    lo, hi = exp_map[seniority]
    years_exp = round(np.random.uniform(lo, hi), 1)

    # Connections correlated with seniority
    conn_map = {
        "entry": (50, 500),
        "mid": (200, 2000),
        "senior": (500, 5000),
        "executive": (2000, 10000),
    }
    lo, hi = conn_map[seniority]
    connections = int(np.random.uniform(lo, hi))

    # Location
    if np.random.random() < 0.6:
        location = f"{np.random.choice(CITIES)}, {np.random.choice(US_STATES)}"
    else:
        location = f"{np.random.choice(CITIES)}, {np.random.choice(COUNTRIES)}"

    # JSON fields
    n_goals = np.random.randint(1, 4)
    n_needs = np.random.randint(1, 4)
    n_offer = np.random.randint(1, 4)
    goals = list(np.random.choice(GOALS_POOL, size=n_goals, replace=False))
    needs = list(np.random.choice(NEEDS_POOL, size=n_needs, replace=False))
    can_offer = list(np.random.choice(OFFER_POOL, size=n_offer, replace=False))

    first = np.random.choice(FIRST_NAMES)
    last = np.random.choice(LAST_NAMES)

    rows.append(
        {
            "profile_id": f"PROF-{i + 1:05d}",
            "name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}{np.random.randint(1, 999)}@email.com",
            "current_role": role,
            "current_company": np.random.choice(COMPANIES),
            "industry": industry,
            "years_experience": years_exp,
            "seniority_level": seniority,
            "skills": str(skills),
            "experience": str(gen_experience(seniority, industry)),
            "education": str(gen_education(seniority)),
            "connections": connections,
            "goals": str(goals),
            "needs": str(needs),
            "can_offer": str(can_offer),
            "location": location,
            "remote_preference": np.random.choice(REMOTE_PREFS),
            "headline": f"{role} at {np.random.choice(COMPANIES)}",
            "about": f"Experienced {role.lower()} with {years_exp} years in {industry.lower()}.",
            "source": "synthetic",
        }
    )

df = pd.DataFrame(rows)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/profiles.csv", index=False)
print(f"Generated {len(df)} profiles")
print(f"Columns: {list(df.columns)}")
print(f"Sample:\n{df.head(2).to_string()}")
