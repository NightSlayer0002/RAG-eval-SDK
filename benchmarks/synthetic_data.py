"""
synthetic_data.py — Generates synthetic RAG evaluation scenarios.

Four categories, 5 scenarios each (20 total, 12 in quick mode):
  CLEAN        — no issues, response correctly grounded
  HALLUCINATED — response contains fabricated information
  CONFLICT     — retrieved chunks contradict each other
  POSITION_BIAS — good middle chunks ignored by LLM
"""

import random


# ======================================================================== #
#  SCENARIO DEFINITIONS
# ======================================================================== #

_CLEAN_SCENARIOS = [
    {
        "id": "clean_1",
        "question": "What are the clinic's opening hours?",
        "context_chunks": [
            {"id": 1, "text": "The clinic is open Monday through Friday from 9 AM to 5 PM."},
            {"id": 2, "text": "We are closed on weekends and public holidays."},
            {"id": 3, "text": "Appointments can be scheduled online or by phone."},
        ],
        "retriever_scores": [0.91, 0.85, 0.72],
        "response": "The clinic is open Monday through Friday from 9 AM to 5 PM and is closed on weekends and public holidays.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "clean_2",
        "question": "Does the clinic offer IVF treatment?",
        "context_chunks": [
            {"id": 1, "text": "We offer a comprehensive range of fertility treatments including IVF, IUI, and ICSI."},
            {"id": 2, "text": "IVF success rates at our clinic are above the national average."},
            {"id": 3, "text": "Our fertility specialists have over 20 years of combined experience."},
        ],
        "retriever_scores": [0.94, 0.89, 0.81],
        "response": "Yes, the clinic offers IVF treatment along with IUI and ICSI. Our success rates are above the national average.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "clean_3",
        "question": "What is the consultation fee?",
        "context_chunks": [
            {"id": 1, "text": "Initial consultation fees are Rs. 1500 for new patients."},
            {"id": 2, "text": "Follow-up consultations are priced at Rs. 800."},
            {"id": 3, "text": "Payment can be made by cash, card, or online transfer."},
        ],
        "retriever_scores": [0.93, 0.88, 0.75],
        "response": "Initial consultations cost Rs. 1500 for new patients. Follow-up visits are Rs. 800. Payment is accepted by cash, card, or online transfer.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "clean_4",
        "question": "Where is the clinic located?",
        "context_chunks": [
            {"id": 1, "text": "Malpani Infertility Clinic is located at Jamuna Nagar, Andheri West, Mumbai."},
            {"id": 2, "text": "The clinic is a 5-minute walk from Andheri station."},
            {"id": 3, "text": "Parking is available in the adjacent building."},
        ],
        "retriever_scores": [0.95, 0.87, 0.79],
        "response": "The clinic is located at Jamuna Nagar, Andheri West, Mumbai, a 5-minute walk from Andheri station. Parking is available nearby.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "clean_5",
        "question": "Who is the doctor at the clinic?",
        "context_chunks": [
            {"id": 1, "text": "Dr. Aniruddha Malpani is the director and chief fertility specialist."},
            {"id": 2, "text": "Dr. Anjali Malpani is a specialist in reproductive medicine and embryology."},
            {"id": 3, "text": "Both doctors have trained at leading institutions in India and abroad."},
        ],
        "retriever_scores": [0.92, 0.90, 0.83],
        "response": "Dr. Aniruddha Malpani is the director and chief fertility specialist. Dr. Anjali Malpani specialises in reproductive medicine.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
]

_HALLUCINATED_SCENARIOS = [
    {
        "id": "halluc_1",
        "question": "What hotels are near the clinic?",
        "context_chunks": [
            {"id": 1, "text": "The clinic is located in Andheri West, a busy commercial area."},
            {"id": 2, "text": "There are several pharmacies within walking distance of the clinic."},
            {"id": 3, "text": "Public transport is easily accessible near the clinic."},
        ],
        "retriever_scores": [0.75, 0.68, 0.61],
        "response": "The following hotels are near the clinic: Hotel Raj Palace, Hotel Blue Nile, and Hotel Woodland Resort.",
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "halluc_2",
        "question": "What is the success rate of IVF?",
        "context_chunks": [
            {"id": 1, "text": "Our clinic has been performing IVF for over two decades."},
            {"id": 2, "text": "We follow the latest evidence-based protocols for fertility treatment."},
            {"id": 3, "text": "Each patient's case is individually assessed for best outcomes."},
        ],
        "retriever_scores": [0.78, 0.72, 0.65],
        "response": "Our IVF success rate is 95% for patients under 35, and we have a 100% embryo survival guarantee.",
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "halluc_3",
        "question": "Does the clinic accept insurance?",
        "context_chunks": [
            {"id": 1, "text": "We offer various payment options to make treatment accessible."},
            {"id": 2, "text": "EMI options are available for long-term treatments."},
            {"id": 3, "text": "Contact our billing department for payment queries."},
        ],
        "retriever_scores": [0.70, 0.63, 0.58],
        "response": "Yes, the clinic accepts all major insurance providers including Star Health, HDFC Ergo, and Bajaj Allianz.",
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "halluc_4",
        "question": "What is the waiting time for appointments?",
        "context_chunks": [
            {"id": 1, "text": "Appointments are available throughout the week."},
            {"id": 2, "text": "We recommend booking in advance for best availability."},
            {"id": 3, "text": "Emergency consultations can sometimes be accommodated."},
        ],
        "retriever_scores": [0.72, 0.65, 0.60],
        "response": "The average waiting time is 3 hours due to high demand. Walk-in patients wait a minimum of 5 hours.",
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "halluc_5",
        "question": "What medications are used in IVF?",
        "context_chunks": [
            {"id": 1, "text": "IVF treatment is customised based on individual patient assessment."},
            {"id": 2, "text": "Our doctors follow international guidelines for fertility protocols."},
            {"id": 3, "text": "All medications are prescribed by qualified fertility specialists."},
        ],
        "retriever_scores": [0.74, 0.66, 0.59],
        "response": "IVF at our clinic uses the revolutionary DragonFire protocol with ExcellaGen hormones and NanoBoost injections.",
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
]

_CONFLICT_SCENARIOS = [
    {
        "id": "conflict_1",
        "question": "Is the clinic open on Saturdays?",
        "context_chunks": [
            {"id": 1, "text": "The clinic is open Monday to Saturday from 9 AM to 6 PM for patient convenience."},
            {"id": 2, "text": "Our working hours are strictly Monday through Friday only. We are closed on weekends."},
            {"id": 3, "text": "For weekend emergencies, please contact the emergency helpline."},
        ],
        "retriever_scores": [0.88, 0.86, 0.71],
        "response": "The clinic is open on Saturdays from 9 AM to 6 PM.",
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "conflict_2",
        "question": "What is the initial consultation fee?",
        "context_chunks": [
            {"id": 1, "text": "New patient consultation fee is Rs. 1500. This includes a full medical history review."},
            {"id": 2, "text": "We charge Rs. 2500 for the initial consultation which includes basic diagnostic tests."},
            {"id": 3, "text": "Payment for consultations is accepted via cash or card."},
        ],
        "retriever_scores": [0.91, 0.89, 0.74],
        "response": "The initial consultation fee is Rs. 1500.",
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "conflict_3",
        "question": "How long does IVF take?",
        "context_chunks": [
            {"id": 1, "text": "A standard IVF cycle takes 4 to 6 weeks from start to embryo transfer."},
            {"id": 2, "text": "The complete IVF process typically spans 2 to 3 months including preparation and recovery."},
            {"id": 3, "text": "Success of IVF depends on multiple individual factors including age and diagnosis."},
        ],
        "retriever_scores": [0.87, 0.85, 0.76],
        "response": "IVF takes approximately 4 to 6 weeks.",
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "conflict_4",
        "question": "Does the clinic offer free consultations?",
        "context_chunks": [
            {"id": 1, "text": "We offer a free 15-minute online consultation for new patients to discuss their case."},
            {"id": 2, "text": "All consultations at the clinic are charged. There are no free consultations available."},
            {"id": 3, "text": "Contact us to learn more about available consultation formats."},
        ],
        "retriever_scores": [0.89, 0.87, 0.72],
        "response": "The clinic offers a free 15-minute online consultation.",
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "conflict_5",
        "question": "How many eggs are typically retrieved during IVF?",
        "context_chunks": [
            {"id": 1, "text": "On average, we retrieve 8 to 12 eggs during an IVF egg collection procedure."},
            {"id": 2, "text": "IVF egg retrieval typically yields 15 to 20 eggs with our stimulation protocol."},
            {"id": 3, "text": "Egg quality is more important than quantity for successful fertilisation."},
        ],
        "retriever_scores": [0.88, 0.86, 0.78],
        "response": "Typically 8 to 12 eggs are retrieved during IVF.",
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
]

# Position bias scenarios: the response only uses first/last chunks,
# ignoring the middle high-relevance chunks
_POSITION_BIAS_SCENARIOS = [
    {
        "id": "bias_1",
        "question": "What does the IVF process involve?",
        "context_chunks": [
            {"id": 1, "text": "IVF begins with ovarian stimulation using hormone injections over 10-14 days."},
            {"id": 2, "text": "Egg retrieval is performed under sedation and takes about 20 minutes."},
            {"id": 3, "text": "Retrieved eggs are fertilised in the laboratory with prepared sperm samples."},
            {"id": 4, "text": "Fertilised embryos are cultured for 3-5 days before transfer."},
            {"id": 5, "text": "Embryo transfer is a painless procedure placing the embryo into the uterus."},
            {"id": 6, "text": "A pregnancy test is taken 14 days after the embryo transfer."},
        ],
        "retriever_scores": [0.88, 0.91, 0.93, 0.90, 0.86, 0.82],
        "response": "IVF begins with hormone injections to stimulate the ovaries. A pregnancy test is done 14 days after transfer.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "bias_2",
        "question": "What are the risks of IVF?",
        "context_chunks": [
            {"id": 1, "text": "The most common IVF risk is Ovarian Hyperstimulation Syndrome (OHSS), affecting about 5% of patients."},
            {"id": 2, "text": "Multiple pregnancies are a significant risk, occurring in about 20-30% of IVF cycles."},
            {"id": 3, "text": "Ectopic pregnancy risk is slightly higher in IVF patients compared to natural conception."},
            {"id": 4, "text": "Egg retrieval carries a small risk of infection or bleeding in rare cases."},
            {"id": 5, "text": "Emotional stress is a recognised side effect of the IVF process for many couples."},
            {"id": 6, "text": "Most IVF risks are manageable with proper medical supervision and follow-up."},
        ],
        "retriever_scores": [0.87, 0.89, 0.91, 0.88, 0.85, 0.80],
        "response": "OHSS affects about 5% of patients. With proper supervision most IVF risks are manageable.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "bias_3",
        "question": "How should I prepare for IVF?",
        "context_chunks": [
            {"id": 1, "text": "Begin taking folic acid supplements at least 3 months before starting IVF treatment."},
            {"id": 2, "text": "Avoid smoking and alcohol completely during the IVF preparation phase."},
            {"id": 3, "text": "Maintain a healthy BMI as obesity can reduce IVF success rates significantly."},
            {"id": 4, "text": "Reduce caffeine intake to less than 200mg per day during IVF treatment."},
            {"id": 5, "text": "Manage stress through yoga, meditation, or counselling during the process."},
            {"id": 6, "text": "Attend all scheduled monitoring appointments for blood tests and ultrasounds."},
        ],
        "retriever_scores": [0.86, 0.90, 0.92, 0.89, 0.87, 0.84],
        "response": "Start folic acid supplements 3 months before IVF. Attend all monitoring appointments for blood tests and ultrasounds.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "bias_4",
        "question": "What happens after egg retrieval in IVF?",
        "context_chunks": [
            {"id": 1, "text": "After egg retrieval, eggs are immediately placed in culture media and assessed for maturity."},
            {"id": 2, "text": "A sperm sample is prepared through washing and selection of the most motile sperm."},
            {"id": 3, "text": "Fertilisation occurs either through conventional IVF mixing or ICSI microinjection."},
            {"id": 4, "text": "Fertilised eggs are monitored daily for cell division and development quality."},
            {"id": 5, "text": "Embryos graded as high quality are selected for transfer or freezing."},
            {"id": 6, "text": "The patient is informed of fertilisation results within 24 hours of egg retrieval."},
        ],
        "retriever_scores": [0.90, 0.88, 0.92, 0.91, 0.89, 0.85],
        "response": "After retrieval, eggs are assessed for maturity and placed in culture. Results are shared within 24 hours.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "bias_5",
        "question": "What support does the clinic offer to IVF patients?",
        "context_chunks": [
            {"id": 1, "text": "We provide a dedicated patient coordinator who guides you through every step of treatment."},
            {"id": 2, "text": "Nutritional counselling is available to optimise diet during treatment."},
            {"id": 3, "text": "Psychological support and counselling sessions are offered to all IVF patients."},
            {"id": 4, "text": "An online patient portal gives you 24/7 access to test results and appointment details."},
            {"id": 5, "text": "Peer support groups connect IVF patients with others on the same journey."},
            {"id": 6, "text": "Our aftercare programme continues support for 6 months following treatment completion."},
        ],
        "retriever_scores": [0.89, 0.87, 0.90, 0.92, 0.88, 0.86],
        "response": "A dedicated patient coordinator guides you through treatment. The aftercare programme provides support for 6 months post-treatment.",
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
]


def build_all_scenarios(quick: bool = False) -> list:
    """
    Build and return all benchmark scenarios.

    Args:
        quick: If True, return 3 per category (12 total) for a fast demo.
               If False, return all 5 per category (20 total).

    Returns:
        List of scenario dicts with question, context_chunks, response,
        retriever_scores, and ground_truth labels.
    """
    n = 3 if quick else 5

    scenarios = []
    for category, pool in [
        ("CLEAN", _CLEAN_SCENARIOS),
        ("HALLUCINATED", _HALLUCINATED_SCENARIOS),
        ("CONFLICT", _CONFLICT_SCENARIOS),
        ("POSITION_BIAS", _POSITION_BIAS_SCENARIOS),
    ]:
        for s in pool[:n]:
            scenario = dict(s)
            scenario["category"] = category
            scenarios.append(scenario)

    return scenarios
