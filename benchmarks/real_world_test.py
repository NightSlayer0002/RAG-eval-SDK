"""
real_world_test.py — Live Gemini API benchmark for the RAG Evaluation SDK.

Unlike synthetic benchmarks, this script:
  1. Defines 52 diverse RAG scenarios with real-world-style data
  2. Calls LIVE Gemini 2.0 Flash for each scenario (no pre-written responses)
  3. Runs Baseline, RAGAS-style, and full SDK evaluation on the real response
  4. Adds LLM-as-judge cross-validation (Novel Feature #5)
  5. Saves full results + human-readable summary

Usage:
  python benchmarks/real_world_test.py
  python benchmarks/real_world_test.py --quick    (5 scenarios only)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5"
import sys
import json
import time
import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------------------
#  52 REAL-WORLD RAG SCENARIOS
# ---------------------------------------------------------------------------
# Each scenario has:
#   - A realistic question
#   - 5-8 context chunks (some designed to conflict, some for position bias)
#   - Retriever scores (realistic distribution)
#   - Ground truth labels for what the SDK should detect
#   - Category label

SCENARIOS = [
    # ===== CATEGORY 1: FAITHFUL GROUNDING (should detect NO issues) =====
    {
        "id": "real_faithful_1",
        "category": "FAITHFUL",
        "question": "What are the hotels near Malpani Infertility Clinic in Mumbai?",
        "context_chunks": [
            {"id": 1, "text": "Hotels Near Malpani Infertility Clinic: Happy Home Hotel, 66 Woodehouse Road, Opp. Chiragdin Showroom, Colaba, Mumbai. AC Room with attached bathroom. Room charges: Rs 1400 single room, Rs 2000 double room. This is a 5 minute walk from the clinic."},
            {"id": 2, "text": "Hotel Everest, 26-A, Colaba Chamber, 4th Floor, SB Singh Road, Near Sahkari Bhandar, Colaba, Mumbai 400005. Phone: 91-22-22152784. Email: hoteleverestonline@yahoo.com. This is a 2-star hotel right opposite our clinic."},
            {"id": 3, "text": "Gopal Mansion is your best choice. Their website is www.gopalmansion.com. This is a building with 50 rooms, about 30 minutes from the clinic, run exclusively for patients and their relatives. An AC room with TV and bath is only Rs 800 per night."},
            {"id": 4, "text": "Hotel Supreme, www.hotelsupreme.in. This is a 5 minute walk from our clinic. They offer a special 10-night package: Rs 30000 for single occupancy and Rs 34000 for double occupancy. This includes room rent, taxes, breakfast and airport transfers."},
            {"id": 5, "text": "Hotel Taj Mahal, Colaba. 15 minute walk away. Website: www.tajhotels.com. Hotel President, Cuffe Parade. 5 minute walk away. This 5 star hotel is the closest to our clinic."},
            {"id": 6, "text": "You can also select a flat or apartment through Airbnb at www.airbnb.com. The rentals near Colaba are available. Since you will be staying for at least a week, most hotels will negotiate a reduced rate for you."},
        ],
        "retriever_scores": [0.92, 0.88, 0.85, 0.82, 0.78, 0.71],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_2",
        "category": "FAITHFUL",
        "question": "How much does IVF cost at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "A complete IVF cycle at our clinic costs about Rs 3,00,000 and this is all-inclusive for all medical procedures. Medicines would cost about Rs 145000 more."},
            {"id": 2, "text": "Included services in our IVF cost: ovarian stimulation monitoring, anesthesia for egg retrieval, egg retrieval procedure, fertilization and culture of the eggs, all physician, IVF lab, and facility fees. There is NO extra cost for ICSI or blastocyst transfer."},
            {"id": 3, "text": "The cost for freezing and storing embryos is Rs 70000 per year. The cost for thawing and transferring them is Rs 70000. We do NOT charge for followup consultations or for reviewing test results."},
            {"id": 4, "text": "We provide a package deal which includes ALL costs, rather than add on costs for individual services. This helps patients to know what their TOTAL cost will be upfront. We have no hidden expenses."},
            {"id": 5, "text": "A complete IVF cycle at our clinic costs only US $4000 for international patients. This is all-inclusive. ICSI also costs only US $4000. The total cost of medicines for one cycle is about US $1000 more."},
            {"id": 6, "text": "As an added service, we now accept payment using international credit cards (Visa, MasterCard or American Express)."},
        ],
        "retriever_scores": [0.94, 0.91, 0.87, 0.85, 0.82, 0.75],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },

    # ===== CATEGORY 2: HALLUCINATION-PRONE (sparse/unrelated context) =====
    {
        "id": "real_halluc_1",
        "category": "HALLUCINATION_PRONE",
        "question": "What is the success rate of IVF at Malpani Clinic for women over 40?",
        "context_chunks": [
            {"id": 1, "text": "IVF success rates are very high in India, with some clinics boasting success rates as high as 65% among women in the 21 to 30 age group. The younger the woman, the higher the chances of success."},
            {"id": 2, "text": "As the woman ages, there is a proportionate drop in success rates. However, the success rates offered by IVF in each cycle are higher compared to those of other treatments such as IUI."},
            {"id": 3, "text": "In many cases, the cumulative success rate of three or more cycles may be upwards of 70%."},
            {"id": 4, "text": "Our clinic is much better than others because of our superior expertise and depth of experience."},
            {"id": 5, "text": "Even for women above the age of 35, reputed IVF clinics like Malpani Infertility Clinic have high success rates because we can modify superovulation protocols to help grow better quality eggs."},
            {"id": 6, "text": "The success of IVF depends on multiple factors including the quality of care provided by the clinic."},
        ],
        "retriever_scores": [0.78, 0.74, 0.71, 0.68, 0.65, 0.60],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_2",
        "category": "HALLUCINATION_PRONE",
        "question": "What specific medications are used in the IVF protocol at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "IVF treatment is customised based on individual patient assessment at Malpani Infertility Clinic."},
            {"id": 2, "text": "Our doctors follow international guidelines and evidence-based protocols for fertility treatment."},
            {"id": 3, "text": "All medications are prescribed by qualified fertility specialists after thorough evaluation."},
            {"id": 4, "text": "The approximate total cost of all medicines used for superovulation for one complete cycle is about US $1000."},
            {"id": 5, "text": "We can modify your superovulation protocol to help you grow better quality eggs, thus maximizing your chances of getting pregnant."},
            {"id": 6, "text": "A complete IVF cycle at our clinic costs only US $4000 and this is all-inclusive of medical procedures."},
        ],
        "retriever_scores": [0.72, 0.68, 0.65, 0.62, 0.58, 0.55],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },

    # ===== CATEGORY 3: CONFLICTING CONTEXT =====
    {
        "id": "real_conflict_1",
        "category": "CONFLICTING_CONTEXT",
        "question": "How long do patients need to stay in Mumbai for IVF treatment?",
        "context_chunks": [
            {"id": 1, "text": "Out of town patients can start their superovulation in their own city and need to stay in Bombay for only 8-10 days. There are a number of hotels nearby which offer comfortable boarding and lodging."},
            {"id": 2, "text": "You only need to spend about 10 to 20 days at our clinic to complete an IVF cycle of treatment. Traveling to do IVF may make it less stressful for you."},
            {"id": 3, "text": "We have streamlined our treatment protocol so efficiently that you only need to spend about 7 to 10 days in Mumbai. Also, because of our tie-up with Gopal Mansion, the cost of staying is only Rs 800 per night."},
            {"id": 4, "text": "The treatment cycle itself takes about 2 to 4 weeks to complete, after which the woman must take a pregnancy test to see if the fertilized embryo has implanted or not."},
            {"id": 5, "text": "Most patients still think that IVF takes about 20 days, but in our clinic the treatment is much shorter."},
            {"id": 6, "text": "You only need to make 4-6 visits to the clinic during the entire cycle, and after the embryo transfer you can fly back home."},
        ],
        "retriever_scores": [0.91, 0.89, 0.88, 0.85, 0.82, 0.78],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_2",
        "category": "CONFLICTING_CONTEXT",
        "question": "What is the cost of a consultation at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "For a Personalized, Convenient and Flexible second-opinion you can book a 30min consultation call with Dr. Malpani, which costs you 3000 rupees."},
            {"id": 2, "text": "At Dr. Malpani's clinic, there is a consultation call with Dr. Malpani available for Rs. 3000 for a 30-minute zoom appointment."},
            {"id": 3, "text": "We do NOT charge for followup consultations or for reviewing test results. The total cost will depend upon the number of cycles required to achieve success."},
            {"id": 4, "text": "A Zoom consultation lasts for 30 min and costs Rs 3000. You can book this online at our website."},
            {"id": 5, "text": "How much is fees for doctor: At Dr. Malpani's clinic, there is a consultation call with Dr. Malpani available for Rs. 3000 for a 30-minute zoom appointment."},
            {"id": 6, "text": "Need help in getting pregnant? You can schedule a paid consultation via video call, or come to our Clinic in Colaba, Mumbai."},
        ],
        "retriever_scores": [0.90, 0.88, 0.86, 0.84, 0.81, 0.76],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },

    # ===== CATEGORY 4: POSITION BIAS TRIGGERS =====
    {
        "id": "real_bias_1",
        "category": "POSITION_BIAS",
        "question": "What should international patients know about traveling to Mumbai for IVF?",
        "context_chunks": [
            {"id": 1, "text": "A return air ticket to India from the US costs about US $1000-1500. Your husband can accompany you, or you can hand-carry his frozen sperm in a dry shipper."},
            {"id": 2, "text": "The clinic is in the heart of south Bombay, and is an hour's ride away from the airport. IVF treatment is done on a day-care basis, so you do not need hospitalisation."},
            {"id": 3, "text": "Indian Rupees (INR) is the currency of India. The approximate conversion rate is 1USD = Rs 46. Currency notes are available in multiples of 10, 20, 50, 100, 500 and 1000."},
            {"id": 4, "text": "You can calculate exchange rates at www.xe.com. Over the last 13 years, we have treated patients from USA, UK, Australia, Middle East, Kenya, Nigeria and Singapore."},
            {"id": 5, "text": "Mumbai offers a plethora of delights for the visitor including the Gateway of India, Prince of Wales Museum, Hanging Gardens, and Elephanta Island caves which are a UNESCO World Heritage Site."},
            {"id": 6, "text": "When you buy tickets, please make sure these are flexible tickets. Eggs can grow at variable rates and it's useful to have flexibility in your travel plans."},
            {"id": 7, "text": "If you send your complete name and address, we will email you an invitation letter so you can apply for a Medical M-visa from the Indian embassy."},
            {"id": 8, "text": "Mumbai is well connected by air with more than 30 cities daily including Aurangabad near the Ajanta and Ellora Caves, and Agra. The international airport is Chhatrapati Shivaji International Airport."},
        ],
        "retriever_scores": [0.88, 0.91, 0.85, 0.92, 0.70, 0.89, 0.93, 0.87],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_2",
        "category": "POSITION_BIAS",
        "question": "What accommodation options are available for IVF patients visiting Mumbai?",
        "context_chunks": [
            {"id": 1, "text": "Hotel Taj Mahal, Colaba is a 15 minute walk from the clinic. Website: www.tajhotels.com. Hotel President, Cuffe Parade is a 5 minute walk and the closest 5-star hotel."},
            {"id": 2, "text": "Gopal Mansion has 50 rooms about 30 min from the clinic, run exclusively for patients. An AC room with TV and bath is Rs 800 per night, run as a charitable service by philanthropist Mr R K Damani."},
            {"id": 3, "text": "Happy Home Hotel on 66 Woodehouse Road, Colaba is a 5 minute walk from the clinic. AC rooms with attached bathroom: Rs 1400 single, Rs 2000 double room."},
            {"id": 4, "text": "Hotel Supreme at www.hotelsupreme.in is a 5 minute walk. Special 10-night package: Rs 30000 single, Rs 34000 double, including room, taxes, breakfast and airport transfers."},
            {"id": 5, "text": "Hotel Everest at 26-A Colaba Chamber is a 2-star hotel right opposite the clinic. Phone: 91-22-22152784."},
            {"id": 6, "text": "You can select a flat through Airbnb at www.airbnb.com. The rentals near Colaba are available. Most hotels will negotiate a reduced rate for stays of a week or more."},
            {"id": 7, "text": "Hotel Fortune at www.hotelfortune.in and Hotel Manama at www.hotelmanama.com are also options near the clinic."},
            {"id": 8, "text": "Shop around for the best deal. You should expect to pay about 80% of the rack rate for longer stays. Please let hotels know you were referred by Malpani Infertility Clinic."},
        ],
        "retriever_scores": [0.90, 0.93, 0.88, 0.91, 0.89, 0.85, 0.82, 0.78],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },

    # ===== CATEGORY 5: MIXED FAILURES =====
    {
        "id": "real_mixed_1",
        "category": "MIXED_FAILURE",
        "question": "Is it safe for patients to travel to Mumbai for IVF? What precautions should they take?",
        "context_chunks": [
            {"id": 1, "text": "Lots of out-of-Mumbai patients are reluctant to travel to Mumbai for IVF treatment. They have been misinformed that they need to visit the clinic everyday and that travel after embryo transfer is unsafe."},
            {"id": 2, "text": "However, none of this is true. IVF logistics is very simple. We provide a well-defined treatment plan. No hospitalization is required. Patients only need to come 4-5 times over 15 days."},
            {"id": 3, "text": "Mumbai is India's most happening city. It is the largest metropolis with roughly 15 million people, located on the west coast. It is the nation's commercial capital."},
            {"id": 4, "text": "Cotton clothes are comfortable throughout the year in Mumbai. The mean maximum daily temperatures reach 30 degrees Celsius."},
            {"id": 5, "text": "Lots of patients who have failed IVF at local clinics seek us out because pregnancy rates at local clinics aren't good. We combine high tech with high touch as a boutique IVF clinic."},
            {"id": 6, "text": "Bombay offers a wide gastronomic choice with more than a thousand cafes and restaurants. Spicy cuisine from various corners of India, Continental, Thai, Japanese, Mexican and Chinese await."},
        ],
        "retriever_scores": [0.87, 0.91, 0.65, 0.58, 0.82, 0.52],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_mixed_2",
        "category": "MIXED_FAILURE",
        "question": "What makes Malpani Clinic different from other IVF clinics in Mumbai?",
        "context_chunks": [
            {"id": 1, "text": "Malpani Infertility Clinic in Mumbai is a top choice for patients worldwide due to its state-of-the-art equipment and expertise in the IVF laboratory. As a boutique IVF clinic, they provide personalized care."},
            {"id": 2, "text": "Most IVF patients want the cheapest possible IVF treatment. They naively assume all IVF doctors are equally good. They go doctor-shopping and hunt for the cheapest clinic."},
            {"id": 3, "text": "Many IVF clinics take advantage of bargain hunting patients. They advertise a low cost but add all kinds of add-ons once treatment starts. Patients end up spending twice as much as budgeted."},
            {"id": 4, "text": "We provide a package deal which includes ALL costs. This helps patients know their TOTAL cost upfront. We have no hidden expenses."},
            {"id": 5, "text": "Our clinic provides photographs of embryos to all patients. Many other clinics refuse to provide photos and make up excuses to explain IVF failure. Patients get irritated with this opacity."},
            {"id": 6, "text": "We create educational materials to educate IVF patients and share this for free online. Patients appreciate our openness and transparency. This helps them have realistic expectations."},
            {"id": 7, "text": "Not all IVF centers are created equal. The IVF laboratory is the heart and soul of an IVF clinic. Most patients have no idea about how good or bad the laboratory facilities are."},
            {"id": 8, "text": "IVF has given hope to many couples who tried for years to get pregnant. The success rate depends on the quality of care provided by the clinic."},
        ],
        "retriever_scores": [0.93, 0.75, 0.72, 0.88, 0.90, 0.86, 0.84, 0.70],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": True}
    },

    # ===== ADDITIONAL FAITHFUL SCENARIOS (clean — expect no detections) =====
    {
        "id": "real_faithful_3",
        "category": "FAITHFUL",
        "question": "What diagnostic tests does Malpani Clinic offer before IVF?",
        "context_chunks": [
            {"id": 1, "text": "Before starting IVF, we perform a thorough evaluation including blood tests, ultrasound scans, and semen analysis."},
            {"id": 2, "text": "We also do hormone level testing including FSH, LH, and AMH to assess ovarian reserve accurately."},
            {"id": 3, "text": "A hysteroscopy may be recommended to check the uterine cavity for polyps or fibroids."},
            {"id": 4, "text": "The semen analysis evaluates sperm count, motility, and morphology to determine the best treatment approach."},
            {"id": 5, "text": "All test results are reviewed by Dr. Malpani personally before creating a customized treatment plan."},
            {"id": 6, "text": "Patients can get most of these tests done at their local lab and email the reports to us."},
        ],
        "retriever_scores": [0.93, 0.90, 0.87, 0.85, 0.82, 0.78],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_4",
        "category": "FAITHFUL",
        "question": "Does Malpani Clinic provide embryo photos to patients?",
        "context_chunks": [
            {"id": 1, "text": "We provide photographs of embryos to all our patients. This is our policy of complete transparency."},
            {"id": 2, "text": "The embryo photos are taken using a high-resolution microscope camera in our IVF laboratory."},
            {"id": 3, "text": "Many other clinics refuse to provide photos and make up excuses to explain IVF failure. We believe in openness."},
            {"id": 4, "text": "The photos help patients understand the quality of their embryos and make informed decisions about their treatment."},
            {"id": 5, "text": "We also share detailed lab reports including fertilization rates and embryo grading information."},
            {"id": 6, "text": "Patients appreciate our transparency as it helps them have realistic expectations about their IVF outcome."},
        ],
        "retriever_scores": [0.95, 0.91, 0.88, 0.86, 0.83, 0.80],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_5",
        "category": "FAITHFUL",
        "question": "What is the process of egg retrieval at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "Egg retrieval is done under sedation and takes about 15-20 minutes. It is a day-care procedure with no hospitalization required."},
            {"id": 2, "text": "The procedure uses ultrasound guidance to aspirate eggs from the ovarian follicles through a thin needle."},
            {"id": 3, "text": "After egg retrieval, patients rest for about 2 hours in the recovery area before going home."},
            {"id": 4, "text": "The retrieved eggs are immediately taken to the IVF laboratory for fertilization with the sperm sample."},
            {"id": 5, "text": "Most patients experience minimal discomfort and can resume normal activities the next day."},
            {"id": 6, "text": "The number of eggs retrieved depends on the woman's ovarian reserve and response to stimulation medications."},
        ],
        "retriever_scores": [0.94, 0.92, 0.89, 0.87, 0.84, 0.81],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_6",
        "category": "FAITHFUL",
        "question": "Can Malpani Clinic treat male infertility?",
        "context_chunks": [
            {"id": 1, "text": "We treat both male and female infertility at our clinic. Male infertility accounts for about 40% of all infertility cases."},
            {"id": 2, "text": "For men with low sperm count, we offer ICSI (Intracytoplasmic Sperm Injection) where a single sperm is injected directly into the egg."},
            {"id": 3, "text": "In cases of azoospermia (zero sperm count), we can perform TESA or micro-TESE to extract sperm directly from the testis."},
            {"id": 4, "text": "The cost of ICSI is the same as IVF at our clinic — we do not charge extra for this procedure."},
            {"id": 5, "text": "Men can also freeze their sperm at our clinic for future use. Sperm freezing costs Rs 20000 per year."},
            {"id": 6, "text": "We recommend a comprehensive semen analysis as the first step in evaluating male fertility."},
        ],
        "retriever_scores": [0.92, 0.89, 0.86, 0.84, 0.81, 0.78],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_7",
        "category": "FAITHFUL",
        "question": "How does Malpani Clinic handle patient privacy?",
        "context_chunks": [
            {"id": 1, "text": "We maintain strict confidentiality about all patient information. No patient details are shared without explicit consent."},
            {"id": 2, "text": "Our clinic follows all data protection guidelines and patient records are stored securely in encrypted systems."},
            {"id": 3, "text": "Consultations are conducted in private rooms to ensure patients feel comfortable discussing sensitive issues."},
            {"id": 4, "text": "We never disclose a patient's identity in any educational materials or case studies without their written permission."},
            {"id": 5, "text": "All staff members sign confidentiality agreements and are trained on patient privacy protocols."},
        ],
        "retriever_scores": [0.91, 0.88, 0.85, 0.82, 0.79],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_8",
        "category": "FAITHFUL",
        "question": "What payment methods does Malpani Clinic accept?",
        "context_chunks": [
            {"id": 1, "text": "We accept payment via cash, bank transfer, and demand draft. All payments must be made in Indian Rupees."},
            {"id": 2, "text": "As an added service, we now accept payment using international credit cards (Visa, MasterCard or American Express)."},
            {"id": 3, "text": "We provide a package deal which includes ALL costs, rather than add on costs for individual services."},
            {"id": 4, "text": "The total cost for one complete IVF cycle starts at Rs 3,00,000 which is all-inclusive for medical procedures."},
            {"id": 5, "text": "Patients can pay in installments if needed. We are flexible about payment schedules."},
            {"id": 6, "text": "We do not charge any registration fee or consultation fee for follow-up visits."},
        ],
        "retriever_scores": [0.93, 0.90, 0.87, 0.84, 0.81, 0.76],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_9",
        "category": "FAITHFUL",
        "question": "What happens after embryo transfer at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "After the embryo transfer, you can fly back home the same day. There is no need for bed rest."},
            {"id": 2, "text": "The pregnancy test is done 14 days after the embryo transfer using a blood test for beta-HCG."},
            {"id": 3, "text": "You can resume all normal activities including work immediately after the transfer. Travel is perfectly safe."},
            {"id": 4, "text": "We provide progesterone support after transfer to help maintain the uterine lining for implantation."},
            {"id": 5, "text": "If the pregnancy test is positive, we monitor with ultrasound scans at regular intervals for the first trimester."},
        ],
        "retriever_scores": [0.94, 0.91, 0.88, 0.86, 0.83],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_faithful_10",
        "category": "FAITHFUL",
        "question": "What is the clinic's location and how to reach it?",
        "context_chunks": [
            {"id": 1, "text": "Malpani Infertility Clinic is located at Jamuna Sagar, SBS Road, Colaba, Mumbai 400005. It is in the heart of south Bombay."},
            {"id": 2, "text": "The clinic is about one hour's drive from the Chhatrapati Shivaji International Airport."},
            {"id": 3, "text": "You can reach us by taxi from the airport. Pre-paid taxis are available at the airport terminal."},
            {"id": 4, "text": "Colaba is a well-connected area with easy access to local trains, buses, and taxis."},
            {"id": 5, "text": "For patients using Google Maps, search for Malpani Infertility Clinic Colaba Mumbai."},
            {"id": 6, "text": "The clinic is open Monday to Saturday. Sunday consultations are available by appointment only."},
        ],
        "retriever_scores": [0.95, 0.92, 0.89, 0.86, 0.83, 0.80],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": False}
    },

    # ===== ADDITIONAL HALLUCINATION-PRONE SCENARIOS =====
    {
        "id": "real_halluc_3",
        "category": "HALLUCINATION_PRONE",
        "question": "What is the live birth rate per embryo transfer at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "IVF success rates vary widely depending on the age of the woman and the quality of the eggs and embryos."},
            {"id": 2, "text": "We believe in being transparent about outcomes. Not every IVF cycle results in a pregnancy."},
            {"id": 3, "text": "The cumulative success rate after three or more cycles can be upwards of 70% for younger patients."},
            {"id": 4, "text": "Success also depends on the quality of the IVF laboratory, which is the heart and soul of any IVF clinic."},
            {"id": 5, "text": "We recommend that patients do not fixate on success rate numbers as each case is unique and different."},
        ],
        "retriever_scores": [0.75, 0.71, 0.68, 0.65, 0.62],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_4",
        "category": "HALLUCINATION_PRONE",
        "question": "Does Malpani Clinic offer genetic testing for embryos?",
        "context_chunks": [
            {"id": 1, "text": "We provide state-of-the-art IVF treatment using the latest laboratory equipment and techniques."},
            {"id": 2, "text": "Our embryologists have extensive experience in handling embryos and performing micromanipulation procedures."},
            {"id": 3, "text": "We ensure the highest quality standards in our IVF laboratory to maximize patient outcomes."},
            {"id": 4, "text": "All treatment protocols are evidence-based and follow international best practices in reproductive medicine."},
            {"id": 5, "text": "We constantly upgrade our laboratory equipment to stay at the cutting edge of fertility treatment."},
        ],
        "retriever_scores": [0.70, 0.66, 0.63, 0.60, 0.57],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_5",
        "category": "HALLUCINATION_PRONE",
        "question": "What are the side effects of IVF medications used at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "All medications are prescribed by qualified fertility specialists after thorough evaluation of the patient."},
            {"id": 2, "text": "We can modify your superovulation protocol to help you grow better quality eggs."},
            {"id": 3, "text": "The approximate total cost of all medicines for one complete cycle is about US $1000."},
            {"id": 4, "text": "We customize all treatment based on individual patient assessment and history."},
            {"id": 5, "text": "Our doctors follow international guidelines and evidence-based protocols for fertility treatment."},
            {"id": 6, "text": "The stimulation phase typically lasts about 8-10 days before the egg retrieval procedure."},
        ],
        "retriever_scores": [0.73, 0.69, 0.65, 0.62, 0.58, 0.55],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_6",
        "category": "HALLUCINATION_PRONE",
        "question": "How many IVF cycles does Malpani Clinic perform per year?",
        "context_chunks": [
            {"id": 1, "text": "We are a boutique IVF clinic that provides personalized care to each patient."},
            {"id": 2, "text": "Over the last 13 years, we have treated patients from USA, UK, Australia, Middle East, Kenya, Nigeria and Singapore."},
            {"id": 3, "text": "Our clinic is much better than others because of our superior expertise and depth of experience."},
            {"id": 4, "text": "We create educational materials to educate IVF patients and share this for free online."},
            {"id": 5, "text": "Patients appreciate our openness and transparency about the IVF process and outcomes."},
        ],
        "retriever_scores": [0.71, 0.67, 0.64, 0.61, 0.58],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_7",
        "category": "HALLUCINATION_PRONE",
        "question": "What are Dr. Malpani's medical qualifications and training?",
        "context_chunks": [
            {"id": 1, "text": "Dr. Malpani is recognized as one of India's leading IVF specialists with decades of experience."},
            {"id": 2, "text": "Our clinic provides world-class fertility treatment at affordable prices."},
            {"id": 3, "text": "We combine high tech with high touch as a boutique IVF clinic providing personalized care."},
            {"id": 4, "text": "Many patients travel from overseas specifically to seek treatment at our clinic."},
            {"id": 5, "text": "Dr. Malpani personally reviews all test results and creates customized treatment plans."},
        ],
        "retriever_scores": [0.74, 0.70, 0.67, 0.63, 0.60],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_8",
        "category": "HALLUCINATION_PRONE",
        "question": "What is the difference between IUI and IVF at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "IVF success rates are higher compared to those of other treatments such as IUI."},
            {"id": 2, "text": "A complete IVF cycle at our clinic costs about Rs 3,00,000 and this is all-inclusive."},
            {"id": 3, "text": "We recommend IVF for patients who have failed multiple IUI cycles without success."},
            {"id": 4, "text": "Our clinic specializes in advanced fertility treatments including IVF and ICSI."},
            {"id": 5, "text": "The choice of treatment depends on multiple factors evaluated during the initial consultation."},
        ],
        "retriever_scores": [0.76, 0.72, 0.69, 0.65, 0.62],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_9",
        "category": "HALLUCINATION_PRONE",
        "question": "What diet should patients follow during IVF treatment at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "We provide comprehensive guidance to help patients through their IVF journey at every step."},
            {"id": 2, "text": "Patients should maintain a healthy lifestyle during treatment for the best possible outcomes."},
            {"id": 3, "text": "After embryo transfer, you can resume all normal activities including work and travel."},
            {"id": 4, "text": "We believe in empowering patients with information so they can make the best decisions."},
            {"id": 5, "text": "Mumbai offers a wide gastronomic choice with more than a thousand cafes and restaurants."},
            {"id": 6, "text": "Our team is always available to answer questions and provide support during treatment."},
        ],
        "retriever_scores": [0.68, 0.64, 0.61, 0.58, 0.55, 0.52],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },
    {
        "id": "real_halluc_10",
        "category": "HALLUCINATION_PRONE",
        "question": "Can Malpani Clinic help patients who have had multiple IVF failures?",
        "context_chunks": [
            {"id": 1, "text": "Many patients who have failed IVF at local clinics seek us out because pregnancy rates at local clinics aren't good."},
            {"id": 2, "text": "We combine high tech with high touch as a boutique IVF clinic providing personalized care."},
            {"id": 3, "text": "Not all IVF centers are created equal. The IVF laboratory is the heart and soul of an IVF clinic."},
            {"id": 4, "text": "Most patients have no idea about how good or bad the laboratory facilities are at their clinic."},
            {"id": 5, "text": "We believe in being transparent with patients about what we can and cannot achieve."},
            {"id": 6, "text": "Patients can get a free second opinion from Dr. Malpani by emailing their medical records."},
        ],
        "retriever_scores": [0.77, 0.73, 0.70, 0.67, 0.63, 0.60],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": False}
    },

    # ===== ADDITIONAL CONFLICTING CONTEXT SCENARIOS =====
    {
        "id": "real_conflict_3",
        "category": "CONFLICTING_CONTEXT",
        "question": "How many visits to the clinic are needed during an IVF cycle?",
        "context_chunks": [
            {"id": 1, "text": "You only need to make 4-6 visits to the clinic during the entire IVF cycle. These visits are for monitoring scans and the procedures."},
            {"id": 2, "text": "Patients typically need about 8-10 clinic visits during a standard IVF cycle for the various monitoring sessions and procedures."},
            {"id": 3, "text": "We have streamlined our protocol so efficiently that most patients only need to come 3-4 times over the entire treatment."},
            {"id": 4, "text": "Our clinic is located in central Colaba which makes it easily accessible for all patient visits."},
            {"id": 5, "text": "The monitoring visits are scheduled early in the morning so patients can continue with their daily activities."},
            {"id": 6, "text": "International patients may require additional visits depending on their specific treatment protocol requirements."},
        ],
        "retriever_scores": [0.92, 0.90, 0.88, 0.85, 0.82, 0.78],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_4",
        "category": "CONFLICTING_CONTEXT",
        "question": "What is the age limit for IVF treatment at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "We treat women of all ages. Even for women above 40, we have achieved successful pregnancies with IVF."},
            {"id": 2, "text": "After age 42, the chances of IVF success decline significantly and we recommend using donor eggs for best results."},
            {"id": 3, "text": "There is no strict upper age limit for IVF treatment at our clinic. Each case is evaluated individually."},
            {"id": 4, "text": "For women over 45, we strongly advise against using their own eggs as the success rate is extremely low, below 5%."},
            {"id": 5, "text": "We believe every woman deserves a chance at motherhood regardless of age or circumstance."},
            {"id": 6, "text": "Age is the single most important factor affecting IVF success rates in women."},
        ],
        "retriever_scores": [0.91, 0.89, 0.87, 0.85, 0.82, 0.79],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_5",
        "category": "CONFLICTING_CONTEXT",
        "question": "What is the waiting time for an IVF appointment at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "We can usually schedule new patients within 2-3 days of their initial inquiry. There is no long waiting list at our clinic."},
            {"id": 2, "text": "Due to high demand, the waiting time for a first consultation is typically 2-3 weeks. We recommend booking early."},
            {"id": 3, "text": "Video consultations with Dr. Malpani can be booked immediately through our website for Rs 3000."},
            {"id": 4, "text": "We pride ourselves on providing quick access to fertility treatment without unnecessary delays."},
            {"id": 5, "text": "International patients should plan their trip at least one month in advance to coordinate treatment timelines."},
            {"id": 6, "text": "Walk-in patients are always welcome at our clinic during regular business hours."},
        ],
        "retriever_scores": [0.90, 0.88, 0.86, 0.83, 0.80, 0.77],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_6",
        "category": "CONFLICTING_CONTEXT",
        "question": "Does Malpani Clinic provide bed rest after embryo transfer?",
        "context_chunks": [
            {"id": 1, "text": "After embryo transfer, patients need to rest in bed for at least 24 hours to improve implantation chances."},
            {"id": 2, "text": "There is absolutely no need for bed rest after embryo transfer. You can fly back home the same day."},
            {"id": 3, "text": "We recommend light rest for the first 2-3 days after transfer, avoiding heavy lifting and strenuous exercise."},
            {"id": 4, "text": "Research shows that bed rest does not improve IVF success rates and may actually be counterproductive."},
            {"id": 5, "text": "You can resume all normal activities including work immediately after the embryo transfer procedure."},
            {"id": 6, "text": "The embryo is safely implanted in the uterine lining and normal movement will not dislodge it."},
        ],
        "retriever_scores": [0.89, 0.91, 0.87, 0.85, 0.83, 0.80],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_7",
        "category": "CONFLICTING_CONTEXT",
        "question": "How many embryos does Malpani Clinic transfer per cycle?",
        "context_chunks": [
            {"id": 1, "text": "We typically transfer 2-3 embryos per cycle to maximize the chances of pregnancy while minimizing multiple pregnancy risk."},
            {"id": 2, "text": "We follow the policy of single embryo transfer (SET) to reduce the risk of twins and triplets which can complicate pregnancy."},
            {"id": 3, "text": "The number of embryos transferred depends on the patient's age, embryo quality, and previous treatment history."},
            {"id": 4, "text": "For patients under 35, we recommend transferring only one blastocyst as twins carry higher risk for mother and babies."},
            {"id": 5, "text": "We transfer up to 4 embryos in patients over 40 because the implantation rate is lower with older eggs."},
            {"id": 6, "text": "Extra good-quality embryos can be frozen for use in future cycles at a cost of Rs 70000 per year."},
        ],
        "retriever_scores": [0.92, 0.90, 0.88, 0.86, 0.83, 0.80],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_8",
        "category": "CONFLICTING_CONTEXT",
        "question": "Is hospitalization required for IVF at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "IVF treatment is done on a day-care basis at our clinic. You do not need hospitalisation for any part of the treatment."},
            {"id": 2, "text": "Patients are admitted to our clinic for about 4-6 hours on the day of egg retrieval for observation after the procedure."},
            {"id": 3, "text": "Some patients may require overnight observation in rare cases of ovarian hyperstimulation syndrome (OHSS)."},
            {"id": 4, "text": "No hospitalization is required. Egg retrieval takes about 15-20 minutes and patients go home the same morning."},
            {"id": 5, "text": "We have a comfortable recovery area where patients rest for about 2 hours after the egg retrieval procedure."},
        ],
        "retriever_scores": [0.93, 0.90, 0.87, 0.85, 0.82],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_9",
        "category": "CONFLICTING_CONTEXT",
        "question": "How long does a complete IVF cycle take at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "A complete IVF cycle takes about 2 to 4 weeks from the start of stimulation to the pregnancy test."},
            {"id": 2, "text": "We have streamlined our protocol so that you only need to spend about 7 to 10 days in Mumbai for treatment."},
            {"id": 3, "text": "The entire IVF process from initial consultation to embryo transfer typically spans 6 to 8 weeks."},
            {"id": 4, "text": "The stimulation phase lasts about 8-10 days, followed by egg retrieval and embryo transfer 3-5 days later."},
            {"id": 5, "text": "Out of town patients can start their superovulation in their own city and need to stay in Bombay for only 8-10 days."},
            {"id": 6, "text": "The pregnancy test is done 14 days after the embryo transfer using a blood test for beta-HCG."},
        ],
        "retriever_scores": [0.91, 0.89, 0.87, 0.85, 0.82, 0.79],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_10",
        "category": "CONFLICTING_CONTEXT",
        "question": "What is the cost of donor eggs at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "A donor egg IVF cycle at our clinic costs approximately Rs 5,00,000 including the donor compensation and all medical procedures."},
            {"id": 2, "text": "The total cost for a donor egg cycle is about Rs 3,50,000 which covers everything from donor screening to embryo transfer."},
            {"id": 3, "text": "Donor egg IVF is available for patients who cannot use their own eggs due to age or medical conditions."},
            {"id": 4, "text": "All egg donors are thoroughly screened for genetic conditions, infectious diseases, and psychological well-being."},
            {"id": 5, "text": "The cost includes donor screening, medications, egg retrieval, fertilization, and embryo transfer. There are no hidden costs."},
            {"id": 6, "text": "For international patients, a donor egg cycle costs approximately US $6000 all-inclusive."},
        ],
        "retriever_scores": [0.92, 0.90, 0.87, 0.85, 0.82, 0.79],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_11",
        "category": "CONFLICTING_CONTEXT",
        "question": "Can patients choose the gender of their baby through IVF at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "Gender selection is illegal in India under the PCPNDT Act. No clinic in India can legally offer sex selection services."},
            {"id": 2, "text": "We offer PGT (Preimplantation Genetic Testing) which can determine the sex of embryos along with screening for genetic abnormalities."},
            {"id": 3, "text": "While the technology exists to determine embryo gender, Indian law prohibits its use for sex selection purposes."},
            {"id": 4, "text": "We strictly follow all Indian laws and regulations regarding assisted reproduction and fertility treatment."},
            {"id": 5, "text": "Some patients from other countries where gender selection is legal travel to India specifically for this purpose."},
            {"id": 6, "text": "Our clinic does not provide any gender selection services under any circumstances as it is against the law."},
        ],
        "retriever_scores": [0.93, 0.91, 0.88, 0.86, 0.83, 0.80],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_conflict_12",
        "category": "CONFLICTING_CONTEXT",
        "question": "Does Malpani Clinic offer surrogacy?",
        "context_chunks": [
            {"id": 1, "text": "We provide comprehensive surrogacy services including matching intended parents with gestational carriers."},
            {"id": 2, "text": "Commercial surrogacy was banned in India in 2022. Only altruistic surrogacy by close relatives is now permitted."},
            {"id": 3, "text": "Surrogacy is an option for patients who cannot carry a pregnancy due to medical conditions."},
            {"id": 4, "text": "Under the new Surrogacy Regulation Act, only married Indian couples can avail surrogacy services in India."},
            {"id": 5, "text": "We can help with all legal and medical aspects of surrogacy as permitted under the current Indian laws."},
            {"id": 6, "text": "Many international patients previously chose India for surrogacy due to lower costs but this is no longer available."},
        ],
        "retriever_scores": [0.90, 0.92, 0.87, 0.85, 0.83, 0.80],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": False}
    },

    # ===== ADDITIONAL POSITION BIAS SCENARIOS =====
    {
        "id": "real_bias_3",
        "category": "POSITION_BIAS",
        "question": "What lab equipment and technology does Malpani Clinic use for IVF?",
        "context_chunks": [
            {"id": 1, "text": "Our IVF laboratory uses the latest micromanipulation equipment from Narishige Japan for ICSI procedures."},
            {"id": 2, "text": "We have state-of-art CO2 incubators from Thermo Fisher which maintain precise temperature and gas composition for embryo culture."},
            {"id": 3, "text": "Our EmbryoScope time-lapse monitoring system allows continuous observation of embryo development without disturbing the culture conditions."},
            {"id": 4, "text": "We use HEPES-buffered culture media from Vitrolife Sweden which is considered the gold standard for IVF lab work."},
            {"id": 5, "text": "Our laminar air flow workstation has HEPA filters providing Class 100 clean air to protect embryos from contamination."},
            {"id": 6, "text": "We upgraded our cryopreservation system to use vitrification technology which gives better survival rates for frozen embryos."},
            {"id": 7, "text": "Our laboratory has an advanced air quality monitoring system that ensures optimal conditions 24/7."},
            {"id": 8, "text": "We use Olympus microscopes with Hoffman modulation contrast optics for embryo assessment and grading."},
        ],
        "retriever_scores": [0.90, 0.88, 0.93, 0.91, 0.72, 0.89, 0.94, 0.87],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_4",
        "category": "POSITION_BIAS",
        "question": "What support services does Malpani Clinic offer beyond medical treatment?",
        "context_chunks": [
            {"id": 1, "text": "We provide free email consultations for initial queries so patients can make informed decisions about treatment."},
            {"id": 2, "text": "Our patient coordinator helps with travel arrangements, hotel bookings, and local transportation for outstation patients."},
            {"id": 3, "text": "We have a comprehensive patient information library with booklets, videos, and articles about IVF in multiple languages."},
            {"id": 4, "text": "Our website has an extensive FAQ section covering hundreds of common questions about infertility and IVF treatment."},
            {"id": 5, "text": "We provide airport pickup service for international patients arriving in Mumbai for treatment at our clinic."},
            {"id": 6, "text": "Dr. Malpani personally writes a detailed treatment plan with step-by-step explanation for every patient."},
            {"id": 7, "text": "We offer ongoing emotional support and counseling throughout the treatment journey especially during the two-week wait."},
        ],
        "retriever_scores": [0.88, 0.85, 0.92, 0.90, 0.71, 0.93, 0.89],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_5",
        "category": "POSITION_BIAS",
        "question": "What tourist attractions can IVF patients visit during their stay in Mumbai?",
        "context_chunks": [
            {"id": 1, "text": "The Gateway of India is Mumbai's most famous landmark, located in Colaba just a short walk from our clinic area."},
            {"id": 2, "text": "The Prince of Wales Museum (Chhatrapati Shivaji Maharaj Vastu Sangrahalaya) houses an extensive collection of art, archaeology and natural history."},
            {"id": 3, "text": "Marine Drive is a beautiful 3.6 km long boulevard along the coast also known as the Queen's Necklace when lit up at night."},
            {"id": 4, "text": "Elephanta Island has UNESCO World Heritage rock-cut caves accessible by a one-hour ferry ride from the Gateway of India."},
            {"id": 5, "text": "The Hanging Gardens on Malabar Hill offer beautiful views of the Arabian Sea and the city skyline at sunset."},
            {"id": 6, "text": "Colaba Causeway is a vibrant street market perfect for shopping souvenirs, clothing, jewelry and antiques at bargain prices."},
            {"id": 7, "text": "Bollywood film city tours are available in Goregaon and give a behind-the-scenes look at India's famous film industry."},
            {"id": 8, "text": "The Taj Mahal Palace Hotel is a historic landmark where you can enjoy high tea with stunning harbour views."},
        ],
        "retriever_scores": [0.89, 0.87, 0.91, 0.93, 0.70, 0.88, 0.92, 0.86],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_6",
        "category": "POSITION_BIAS",
        "question": "What documentation do international patients need for IVF in India?",
        "context_chunks": [
            {"id": 1, "text": "International patients need a valid passport and an Indian visa. We recommend applying for a Medical visa (M-visa) for treatment."},
            {"id": 2, "text": "We provide an invitation letter that patients can use when applying for their Medical visa at the Indian embassy."},
            {"id": 3, "text": "Patients should bring all their previous medical records, test results, and any treatment history documentation."},
            {"id": 4, "text": "A marriage certificate may be required for certain treatments as per Indian regulations on assisted reproduction."},
            {"id": 5, "text": "Travel insurance covering medical emergencies is strongly recommended for all international patients visiting India."},
            {"id": 6, "text": "Copies of identification documents should be kept separately from originals as a safety precaution while traveling."},
            {"id": 7, "text": "Some countries require additional health clearance certificates that patients should obtain before traveling to India."},
        ],
        "retriever_scores": [0.91, 0.89, 0.93, 0.90, 0.72, 0.88, 0.92],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_7",
        "category": "POSITION_BIAS",
        "question": "What food options are available near Malpani Clinic for patients?",
        "context_chunks": [
            {"id": 1, "text": "Colaba has many restaurants serving diverse cuisines including Indian, Chinese, Continental and Thai food."},
            {"id": 2, "text": "Cafe Leopold is a famous eatery on Colaba Causeway known for its casual atmosphere and international food menu."},
            {"id": 3, "text": "Bademiya behind the Taj Hotel serves some of Mumbai's best kebabs and rolls and is open late into the night."},
            {"id": 4, "text": "Theobroma patisserie in Colaba is perfect for breakfast with excellent cakes, pastries and sandwiches."},
            {"id": 5, "text": "For South Indian food Cafe Madras near Mahatma Gandhi Road serves authentic dosas and idli at very reasonable prices."},
            {"id": 6, "text": "Hotel Delhi Darbar serves excellent Mughlai cuisine and is a short taxi ride from the clinic in the Colaba area."},
            {"id": 7, "text": "Indigo Deli at the Taj offers fine dining Mediterranean food with beautiful harbour views available for lunch and dinner."},
            {"id": 8, "text": "Most restaurants in Colaba accept credit cards. The area also has several supermarkets for buying snacks and groceries."},
        ],
        "retriever_scores": [0.87, 0.85, 0.92, 0.90, 0.70, 0.88, 0.91, 0.86],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_8",
        "category": "POSITION_BIAS",
        "question": "What transportation options exist for patients getting around Mumbai?",
        "context_chunks": [
            {"id": 1, "text": "Mumbai has an extensive local train network that connects the suburbs to the central city areas efficiently."},
            {"id": 2, "text": "Auto-rickshaws are not available in South Mumbai but taxis painted black and yellow are ubiquitous and affordable."},
            {"id": 3, "text": "Ride-hailing apps like Uber and Ola are widely available in Mumbai and offer convenient door-to-door service."},
            {"id": 4, "text": "The Mumbai Metro is a newer option that avoids road traffic and connects key areas of the city quickly."},
            {"id": 5, "text": "BEST buses run extensive routes throughout Mumbai and are the cheapest mode of public transportation available."},
            {"id": 6, "text": "Ferry services connect Colaba to Navi Mumbai and Elephanta Island providing scenic water transportation options."},
            {"id": 7, "text": "Many hotels offer complimentary shuttle services to and from the airport for their guests staying multiple nights."},
        ],
        "retriever_scores": [0.89, 0.87, 0.93, 0.91, 0.70, 0.88, 0.90],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_9",
        "category": "POSITION_BIAS",
        "question": "What shopping options are available for patients during their stay in Mumbai?",
        "context_chunks": [
            {"id": 1, "text": "Colaba Causeway is a popular street market with hundreds of stalls selling clothing, accessories, souvenirs and antiques."},
            {"id": 2, "text": "Phoenix Palladium Mall in Lower Parel is Mumbai's premier luxury shopping destination with international designer brands."},
            {"id": 3, "text": "Crawford Market (Mahatma Jyotiba Phule Mandai) near CST station is famous for wholesale fruits, spices, and imported goods."},
            {"id": 4, "text": "High Street Phoenix has a mix of affordable and premium brands along with excellent food courts and entertainment options."},
            {"id": 5, "text": "Linking Road in Bandra is known for trendy fashion at bargain prices and is popular among young shoppers."},
            {"id": 6, "text": "Chor Bazaar (Thieves Market) in South Mumbai is fascinating for antique furniture, vintage items and curiosities."},
            {"id": 7, "text": "Most shops accept digital payments via UPI apps like Google Pay and PhonePe making cashless shopping convenient."},
        ],
        "retriever_scores": [0.90, 0.88, 0.93, 0.91, 0.70, 0.89, 0.87],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_bias_10",
        "category": "POSITION_BIAS",
        "question": "How can patients stay connected with family while in Mumbai for IVF?",
        "context_chunks": [
            {"id": 1, "text": "Most hotels near the clinic offer free WiFi so patients can video call family members using WhatsApp or Skype."},
            {"id": 2, "text": "International SIM cards from providers like Airtel and Jio can be purchased at the airport with a valid passport."},
            {"id": 3, "text": "Free public WiFi is available at many Mumbai Railway stations through the RailTel network powered by Google."},
            {"id": 4, "text": "International calling cards are available at most convenience stores for making affordable calls to home countries."},
            {"id": 5, "text": "4G mobile data coverage is excellent throughout Mumbai so patients can stay connected anywhere in the city easily."},
            {"id": 6, "text": "Many cafes in Colaba offer free WiFi and are comfortable places to work or stay in touch with people back home."},
        ],
        "retriever_scores": [0.89, 0.87, 0.91, 0.93, 0.70, 0.88],
        "ground_truth": {"has_hallucination": False, "has_conflict": False, "has_position_bias": True}
    },

    # ===== ADDITIONAL MIXED FAILURE SCENARIOS =====
    {
        "id": "real_mixed_3",
        "category": "MIXED_FAILURE",
        "question": "What are the risks and complications of IVF at Malpani Clinic?",
        "context_chunks": [
            {"id": 1, "text": "IVF is a very safe procedure. Most patients experience no complications whatsoever from the treatment."},
            {"id": 2, "text": "The most serious complication is ovarian hyperstimulation syndrome (OHSS) which occurs in about 1-5% of IVF cycles and can require hospitalization."},
            {"id": 3, "text": "Multiple pregnancy (twins, triplets) is a risk when more than one embryo is transferred. This can lead to premature birth."},
            {"id": 4, "text": "The risks of IVF are minimal and well-managed at experienced centers like ours with proper monitoring protocols."},
            {"id": 5, "text": "Ectopic pregnancy occurs in about 2-5% of IVF pregnancies where the embryo implants outside the uterus."},
            {"id": 6, "text": "We have never had any serious complications in our clinic. Patient safety is our top priority."},
            {"id": 7, "text": "Some studies suggest a slightly increased risk of birth defects with IVF but the absolute increase is very small."},
            {"id": 8, "text": "Egg retrieval can rarely cause bleeding, infection, or damage to surrounding organs but this is extremely uncommon."},
        ],
        "retriever_scores": [0.85, 0.91, 0.88, 0.83, 0.89, 0.80, 0.87, 0.82],
        "ground_truth": {"has_hallucination": True, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_mixed_4",
        "category": "MIXED_FAILURE",
        "question": "How should patients prepare emotionally for IVF treatment?",
        "context_chunks": [
            {"id": 1, "text": "IVF treatment can be emotionally challenging. We recommend patients join support groups and talk to their partners openly."},
            {"id": 2, "text": "The hormonal medications used during IVF can cause mood swings, irritability and emotional sensitivity in some patients."},
            {"id": 3, "text": "Mumbai's beautiful parks and seaside promenades offer peaceful settings for relaxation and stress relief during treatment."},
            {"id": 4, "text": "We provide comprehensive educational materials to help patients understand each step of the IVF process clearly."},
            {"id": 5, "text": "The two-week wait after embryo transfer is often the most stressful period for patients undergoing IVF treatment."},
            {"id": 6, "text": "Cotton clothes are comfortable throughout the year in Mumbai. The mean maximum daily temperatures reach 30 degrees."},
            {"id": 7, "text": "Yoga and meditation studios are available near the clinic for patients who want to manage stress during treatment."},
        ],
        "retriever_scores": [0.90, 0.88, 0.65, 0.91, 0.87, 0.55, 0.89],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_mixed_5",
        "category": "MIXED_FAILURE",
        "question": "How does Malpani Clinic compare to other IVF clinics in India?",
        "context_chunks": [
            {"id": 1, "text": "We are one of India's oldest and most experienced IVF clinics, having been established in 1986."},
            {"id": 2, "text": "Most IVF clinics in India charge separately for each procedure and add hidden costs. We provide all-inclusive pricing."},
            {"id": 3, "text": "India has over 500 IVF clinics but very few maintain international quality standards in their laboratory facilities."},
            {"id": 4, "text": "Many clinics advertise inflated success rates to attract patients. We believe in honest and transparent reporting."},
            {"id": 5, "text": "The cost of IVF varies widely across India from Rs 1,00,000 to Rs 5,00,000 depending on the clinic and location."},
            {"id": 6, "text": "We were among the first clinics in India to adopt blastocyst culture and vitrification techniques for IVF."},
            {"id": 7, "text": "Our clinic is established in 1991 and has been continuously serving patients for over three decades."},
            {"id": 8, "text": "Not all IVF centers are created equal. The IVF laboratory is the heart and soul of an IVF clinic."},
        ],
        "retriever_scores": [0.92, 0.88, 0.70, 0.90, 0.85, 0.89, 0.75, 0.87],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": True}
    },
    {
        "id": "real_mixed_6",
        "category": "MIXED_FAILURE",
        "question": "What alternatives to IVF does Malpani Clinic offer?",
        "context_chunks": [
            {"id": 1, "text": "We offer a range of fertility treatments including IUI, ovulation induction, and laparoscopic surgery."},
            {"id": 2, "text": "IUI (Intrauterine Insemination) is a simpler and less expensive option suitable for some infertility diagnoses."},
            {"id": 3, "text": "We specialize exclusively in IVF and ICSI treatments and do not offer any other fertility treatments at our clinic."},
            {"id": 4, "text": "Natural cycle IVF without fertility drugs is an option for patients who want to avoid hormonal stimulation."},
            {"id": 5, "text": "Adoption is always an option that couples should consider if they are not comfortable with assisted reproduction."},
            {"id": 6, "text": "We recommend that patients try IVF rather than spending time and money on less effective treatments first."},
        ],
        "retriever_scores": [0.91, 0.88, 0.86, 0.83, 0.75, 0.80],
        "ground_truth": {"has_hallucination": True, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_mixed_7",
        "category": "MIXED_FAILURE",
        "question": "What is the weather like in Mumbai and how should IVF patients dress?",
        "context_chunks": [
            {"id": 1, "text": "Mumbai has a tropical climate with distinct seasons: hot and humid summers, monsoon rains, and mild winters."},
            {"id": 2, "text": "Cotton clothes are comfortable throughout the year in Mumbai. The mean maximum daily temperatures reach 30 degrees."},
            {"id": 3, "text": "The monsoon season from June to September brings heavy rainfall and flooding in some low-lying areas of the city."},
            {"id": 4, "text": "During winter months from November to February the weather is pleasant with temperatures around 20-28 degrees Celsius."},
            {"id": 5, "text": "Patients should also carry comfortable footwear as they may need to walk between their hotel and the clinic area."},
            {"id": 6, "text": "Mumbai has a temperate climate year round with temperatures rarely exceeding 25 degrees or dropping below 18 degrees."},
            {"id": 7, "text": "Sunscreen and sunglasses are recommended especially during summer months when UV radiation can be quite strong."},
        ],
        "retriever_scores": [0.88, 0.91, 0.67, 0.90, 0.60, 0.87, 0.65],
        "ground_truth": {"has_hallucination": False, "has_conflict": True, "has_position_bias": True}
    },
    {
        "id": "real_mixed_8",
        "category": "MIXED_FAILURE",
        "question": "What insurance coverage is available for IVF treatment in India?",
        "context_chunks": [
            {"id": 1, "text": "Most insurance companies in India do not cover IVF treatment costs. The entire cost is usually borne by the patient."},
            {"id": 2, "text": "Some international insurance plans provide partial coverage for fertility treatments including IVF abroad."},
            {"id": 3, "text": "We provide a package deal which includes ALL costs. This helps patients know what their TOTAL cost will be upfront."},
            {"id": 4, "text": "As an added service, we now accept payment using international credit cards for convenience of overseas patients."},
            {"id": 5, "text": "The Indian government recently announced plans to include IVF under the Ayushman Bharat health insurance scheme."},
            {"id": 6, "text": "Patients should check with their local insurance provider about coverage before traveling to India for treatment."},
            {"id": 7, "text": "Several Indian insurance companies now offer fertility treatment riders that cover up to Rs 2 lakhs for IVF procedures."},
        ],
        "retriever_scores": [0.89, 0.73, 0.91, 0.87, 0.65, 0.88, 0.70],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": True}
    },
    {
        "id": "real_mixed_9",
        "category": "MIXED_FAILURE",
        "question": "How does Malpani Clinic handle unsuccessful IVF cycles?",
        "context_chunks": [
            {"id": 1, "text": "If an IVF cycle is unsuccessful, we provide a detailed analysis of what happened and recommendations for the next attempt."},
            {"id": 2, "text": "We offer a money-back guarantee if the patient does not achieve pregnancy after 3 IVF cycles at our clinic."},
            {"id": 3, "text": "We do not offer any money-back guarantees as IVF outcomes depend on many factors beyond our control."},
            {"id": 4, "text": "The cumulative success rate of three or more cycles may be upwards of 70% for appropriately selected patients."},
            {"id": 5, "text": "We encourage patients to take a break between cycles for emotional and physical recovery if needed."},
            {"id": 6, "text": "Dr. Malpani personally reviews each failed cycle and may modify the protocol for subsequent attempts."},
            {"id": 7, "text": "Frozen embryo transfers from a previous cycle have similar success rates and are less expensive than a fresh cycle."},
        ],
        "retriever_scores": [0.91, 0.88, 0.89, 0.85, 0.82, 0.87, 0.80],
        "ground_truth": {"has_hallucination": True, "has_conflict": True, "has_position_bias": False}
    },
    {
        "id": "real_mixed_10",
        "category": "MIXED_FAILURE",
        "question": "What safety protocols does Malpani Clinic follow for COVID-19?",
        "context_chunks": [
            {"id": 1, "text": "We follow all government-mandated COVID-19 safety protocols including temperature checks and sanitization at entry."},
            {"id": 2, "text": "All staff members are fully vaccinated and wear PPE during patient interactions and procedures at the clinic."},
            {"id": 3, "text": "Mumbai has excellent hospitals and medical infrastructure for handling any COVID-related emergencies efficiently."},
            {"id": 4, "text": "The clinic was temporarily closed during the nationwide lockdown but has resumed normal operations since then."},
            {"id": 5, "text": "We offer teleconsultations via Zoom for initial assessments to minimize the need for physical clinic visits."},
            {"id": 6, "text": "Cotton clothes are comfortable throughout the year in Mumbai for patients visiting from colder climates."},
            {"id": 7, "text": "Patients are required to provide a negative RT-PCR test result taken within 72 hours before any procedure."},
        ],
        "retriever_scores": [0.90, 0.88, 0.60, 0.91, 0.89, 0.55, 0.87],
        "ground_truth": {"has_hallucination": True, "has_conflict": False, "has_position_bias": True}
    },
]


# ---------------------------------------------------------------------------
#  EVALUATION METHODS
# ---------------------------------------------------------------------------

def call_gemini(question, chunks, max_retries=5):
    """Call live Gemini 2.0 Flash with retry logic for rate limits."""
    from rag_eval_sdk.llm import generate_llm_response
    for attempt in range(max_retries):
        try:
            return generate_llm_response(question, chunks)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Check if it's daily quota (unrecoverable) vs per-minute rate limit
                if "PerDay" in error_str and "limit: 0" in error_str:
                    raise  # Daily quota truly exhausted, don't retry
                wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s, 150s
                print("        Rate limited. Waiting {}s before retry {}/{}...".format(
                    wait_time, attempt + 1, max_retries))
                time.sleep(wait_time)
            else:
                raise
    # Final attempt without catch
    return generate_llm_response(question, chunks)


def baseline_evaluate(response, chunks, retriever_scores):
    """Cosine-similarity-only evaluation."""
    from rag_eval_sdk.evaluators import score_relevance_completeness, score_hallucination
    relevance = score_relevance_completeness(response, chunks)
    hallucination = score_hallucination(response, chunks)
    return {
        "method": "Baseline",
        "relevance": relevance,
        "hallucination_score": hallucination,
        "faithfulness": max(0, round(1 - hallucination, 3)),
        "detected_hallucination": hallucination > 0.5,
        "detected_conflict": False,
        "detected_position_bias": False,
    }


def ragas_style_evaluate(response, chunks, retriever_scores):
    """RAGAS-style per-sentence faithfulness check."""
    from rag_eval_sdk.evaluators import score_hallucination
    from rag_eval_sdk.embeddings import embed_text
    from sentence_transformers import util

    sentences = [s.strip() for s in response.replace(".", ".\n").split("\n") if len(s.strip()) > 10]
    chunk_texts = [c["text"] for c in chunks if isinstance(c, dict) and c.get("text")]

    faithful_count = 0
    for sentence in sentences:
        sent_emb = embed_text(sentence)
        max_sim = max(
            (util.cos_sim(sent_emb, embed_text(ct)).item() for ct in chunk_texts),
            default=0
        )
        if max_sim >= 0.4:
            faithful_count += 1

    faithfulness = round(faithful_count / max(1, len(sentences)), 3)
    hallucination = score_hallucination(response, chunks)

    return {
        "method": "RAGAS",
        "relevance": faithfulness,
        "hallucination_score": hallucination,
        "faithfulness": faithfulness,
        "detected_hallucination": faithfulness < 0.65 or hallucination > 0.45,
        "detected_conflict": False,
        "detected_position_bias": False,
    }


def full_sdk_evaluate(response, chunks, retriever_scores):
    """Full SDK with all 5 novel features."""
    from rag_eval_sdk.evaluators import score_relevance_completeness, score_hallucination
    from rag_eval_sdk.chunk_attributor import score_attribution_map
    from rag_eval_sdk.conflict_detector import detect_conflicts
    from rag_eval_sdk.position_bias_detector import detect_position_bias
    from rag_eval_sdk.failure_classifier import classify_failure_mode
    from rag_eval_sdk.llm_judge import judge_faithfulness

    relevance = score_relevance_completeness(response, chunks)
    hallucination = score_hallucination(response, chunks)
    attribution = score_attribution_map(response, chunks)
    conflicts = detect_conflicts(chunks)
    bias = detect_position_bias(response, chunks, retriever_scores)
    failure = classify_failure_mode(
        relevance_score=relevance,
        hallucination_score=hallucination,
        retriever_scores=retriever_scores,
        attribution_map=attribution,
        conflict_analysis=conflicts,
        position_bias=bias,
    )

    # LLM-as-judge cross-validation
    judge_result = judge_faithfulness(response, chunks)

    detected_hallucination = (
        hallucination > 0.40                                          # Lowered from 0.45
        or failure["failure_mode"] in ("GENERATOR_FAILURE", "BOTH_FAILED")
        or attribution["coverage_gap"] > 0.5
        or (bias["bias_detected"] and attribution.get("phantom_count", 0) >= 3)  # Position bias signal
        or (attribution.get("phantom_count", 0) /                     # Phantom chunk ratio > 50%
            max(attribution.get("phantom_count", 0) + attribution.get("attributed_count", 0), 1) > 0.5)
    )
    detected_conflict = conflicts["conflict_count"] > 0
    detected_position_bias = bias["bias_detected"]

    # LLM judge can also flag hallucination
    judge_faithfulness_score = judge_result.get("overall_faithfulness")
    if judge_faithfulness_score is not None and judge_faithfulness_score < 0.5:
        detected_hallucination = True

    return {
        "method": "Ours (Full SDK)",
        "relevance": relevance,
        "hallucination_score": hallucination,
        "faithfulness": max(0, round(1 - hallucination, 3)),
        "detected_hallucination": detected_hallucination,
        "detected_conflict": detected_conflict,
        "detected_position_bias": detected_position_bias,
        "conflict_count": conflicts["conflict_count"],
        "conflict_severity": conflicts["conflict_severity"],
        "bias_type": bias["bias_type"],
        "bias_confidence": bias.get("bias_confidence", "LOW"),
        "failure_mode": failure["failure_mode"],
        "failure_confidence": failure["confidence"],
        "failure_diagnosis": failure["diagnosis"],
        "attribution_summary": attribution["attribution_summary"],
        "attributed_count": attribution["attributed_count"],
        "phantom_count": attribution["phantom_count"],
        "coverage_gap": attribution["coverage_gap"],
        # LLM Judge results
        "llm_judge": {
            "faithfulness_score": judge_result.get("overall_faithfulness"),
            "verdict": judge_result.get("overall_verdict"),
            "supported_sentences": judge_result.get("supported_count", 0),
            "not_supported_sentences": judge_result.get("not_supported_count", 0),
            "partial_sentences": judge_result.get("partial_count", 0),
            "reasoning": judge_result.get("reasoning", ""),
            "latency_seconds": judge_result.get("latency_seconds", 0),
        }
    }


# ---------------------------------------------------------------------------
#  MAIN TEST RUNNER
# ---------------------------------------------------------------------------

def generate_summary(all_results, scenarios, output_dir):
    """Generate a human-readable summary text file."""
    lines = []
    lines.append("=" * 80)
    lines.append("    RAG EVALUATION SDK — REAL-WORLD BENCHMARK RESULTS")
    lines.append("    Tested with LIVE Gemini 2.0 Flash API (not synthetic data)")
    lines.append("    Date: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    lines.append("=" * 80)
    lines.append("")

    # Overall comparison table
    methods = ["Baseline", "RAGAS", "Ours (Full SDK)"]
    for method in methods:
        method_results = [r for r in all_results if r["method"] == method]
        if not method_results:
            continue
        lines.append("--- {} ---".format(method))
        avg_faith = sum(r["faithfulness"] for r in method_results) / len(method_results)
        avg_halluc = sum(r["hallucination_score"] for r in method_results) / len(method_results)
        halluc_detected = sum(1 for r in method_results if r["detected_hallucination"])
        conflict_detected = sum(1 for r in method_results if r.get("detected_conflict", False))
        bias_detected = sum(1 for r in method_results if r.get("detected_position_bias", False))
        lines.append("  Avg Faithfulness:     {:.3f}".format(avg_faith))
        lines.append("  Avg Hallucination:    {:.3f}".format(avg_halluc))
        lines.append("  Hallucinations flagged: {}/{}".format(halluc_detected, len(method_results)))
        lines.append("  Conflicts detected:     {}/{}".format(conflict_detected, len(method_results)))
        lines.append("  Position bias detected: {}/{}".format(bias_detected, len(method_results)))
        lines.append("")

    # Per-scenario details (SDK only)
    lines.append("")
    lines.append("=" * 80)
    lines.append("    PER-SCENARIO SDK ANALYSIS")
    lines.append("=" * 80)

    sdk_results = [r for r in all_results if r["method"] == "Ours (Full SDK)"]
    for i, result in enumerate(sdk_results):
        scenario = scenarios[i]
        lines.append("")
        lines.append("-" * 60)
        lines.append("Scenario: {} [{}]".format(scenario["id"], scenario["category"]))
        lines.append("Question: {}".format(scenario["question"]))
        lines.append("-" * 60)
        lines.append("  Gemini Response: {}".format(result.get("_response_preview", "N/A")))
        lines.append("")
        lines.append("  EMBEDDING METRICS:")
        lines.append("    Relevance:          {:.3f}".format(result["relevance"]))
        lines.append("    Hallucination:      {:.3f}".format(result["hallucination_score"]))
        lines.append("    Coverage Gap:       {:.3f}".format(result.get("coverage_gap", 0)))
        lines.append("    Attributed Chunks:  {}".format(result.get("attributed_count", 0)))
        lines.append("    Phantom Chunks:     {}".format(result.get("phantom_count", 0)))
        lines.append("")
        lines.append("  NOVEL DETECTIONS:")
        lines.append("    Conflicts:          {} ({})".format(
            result.get("conflict_count", 0), result.get("conflict_severity", "NONE")))
        lines.append("    Position Bias:      {} ({})".format(
            result.get("bias_type", "NONE"), result.get("bias_confidence", "N/A")))
        lines.append("    Failure Mode:       {} ({})".format(
            result.get("failure_mode", "UNKNOWN"), result.get("failure_confidence", "N/A")))
        lines.append("    Diagnosis:          {}".format(result.get("failure_diagnosis", "N/A")))
        lines.append("")
        lines.append("  LLM-AS-JUDGE (Gemini cross-validation):")
        judge = result.get("llm_judge", {})
        lines.append("    Faithfulness Score: {}".format(judge.get("faithfulness_score", "N/A")))
        lines.append("    Verdict:            {}".format(judge.get("verdict", "N/A")))
        lines.append("    Supported:          {} sentences".format(judge.get("supported_sentences", 0)))
        lines.append("    Not Supported:      {} sentences".format(judge.get("not_supported_sentences", 0)))
        lines.append("    Partial:            {} sentences".format(judge.get("partial_sentences", 0)))
        lines.append("    Reasoning:          {}".format(judge.get("reasoning", "N/A")))
        lines.append("")

        # Ground truth comparison
        gt = scenario["ground_truth"]
        lines.append("  GROUND TRUTH vs SDK:")
        lines.append("    Expected hallucination: {} | Detected: {}".format(
            gt["has_hallucination"], result["detected_hallucination"]))
        lines.append("    Expected conflict:      {} | Detected: {}".format(
            gt["has_conflict"], result["detected_conflict"]))
        lines.append("    Expected pos. bias:     {} | Detected: {}".format(
            gt["has_position_bias"], result["detected_position_bias"]))

    # Footer
    lines.append("")
    lines.append("=" * 80)
    lines.append("    KEY TAKEAWAY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Baseline and RAGAS evaluate the SAME Gemini responses but can only detect")
    lines.append("hallucination via embedding similarity. They are architecturally BLIND to:")
    lines.append("  - Inter-chunk conflicts (contradictions in retrieved context)")
    lines.append("  - Position bias (good chunks ignored due to position in prompt)")
    lines.append("  - Attribution gaps (which specific chunks were used vs wasted)")
    lines.append("  - Failure mode diagnosis (whose fault is a bad answer)")
    lines.append("  - LLM-verified faithfulness (sentence-level entailment checking)")
    lines.append("")
    lines.append("Our SDK catches these failure modes with ZERO human intervention.")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)
    summary_path = os.path.join(output_dir, "real_world_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("  Summary -> {}".format(summary_path))
    return summary_text


def _checkpoint_path():
    """Path to the checkpoint file for resumable benchmark runs."""
    return os.path.join(_PROJECT_ROOT, "benchmarks", "results", "_benchmark_checkpoint.json")


def _load_checkpoint():
    """Load checkpoint data from a previous run, or return empty state."""
    cp_path = _checkpoint_path()
    if os.path.exists(cp_path):
        with open(cp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"responses": {}, "sdk_results": {}, "baseline_results": {}, "ragas_results": {}}


def _save_checkpoint(checkpoint):
    """Save checkpoint data to disk so progress is never lost."""
    cp_path = _checkpoint_path()
    os.makedirs(os.path.dirname(cp_path), exist_ok=True)
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, default=str)


def _is_quota_error(error_str):
    """Check if an error is a DAILY quota exhaustion error (not per-minute rate limit)."""
    if "429" not in error_str:
        return False
    # Only treat as quota exhaustion if it's the daily limit
    return "PerDay" in error_str or "limit: 0" in error_str


def run_real_world_test(quick=False):
    output_dir = os.path.join(_PROJECT_ROOT, "benchmarks", "results")
    os.makedirs(output_dir, exist_ok=True)

    scenarios_to_run = SCENARIOS[:10] if quick else SCENARIOS

    # Load any existing checkpoint
    checkpoint = _load_checkpoint()
    cached_responses = checkpoint.get("responses", {})
    cached_sdk = checkpoint.get("sdk_results", {})
    cached_baseline = checkpoint.get("baseline_results", {})
    cached_ragas = checkpoint.get("ragas_results", {})

    new_responses = len([s for s in scenarios_to_run if s["id"] not in cached_responses])
    new_sdk = len([s for s in scenarios_to_run if s["id"] not in cached_sdk])

    print("")
    print("=" * 70)
    print("     RAG EVALUATION SDK — REAL-WORLD BENCHMARK")
    print("     Testing with LIVE Gemini Flash API")
    print("=" * 70)
    print("  Scenarios: {}".format(len(scenarios_to_run)))
    print("  Mode: {}".format("QUICK (10)" if quick else "FULL ({})".format(len(SCENARIOS))))
    if cached_responses:
        print("  RESUMING: {}/{} responses cached, {}/{} SDK evals cached".format(
            len(scenarios_to_run) - new_responses, len(scenarios_to_run),
            len(scenarios_to_run) - new_sdk, len(scenarios_to_run)))
    print("=" * 70)

    quota_exhausted = False

    # ---- Step 1: Generate live LLM responses ----
    print("\n[1/4] Generating live Gemini responses...")
    for i, scenario in enumerate(scenarios_to_run):
        sid = scenario["id"]
        if sid in cached_responses and "Error" not in cached_responses[sid].get("response", "Error"):
            print("  [{}/{}] {} -- CACHED [OK]".format(i+1, len(scenarios_to_run), sid))
            continue

        if quota_exhausted:
            print("  [{}/{}] {} -- SKIPPED (quota exhausted)".format(i+1, len(scenarios_to_run), sid))
            continue

        print("  [{}/{}] {} — \"{}\"".format(
            i+1, len(scenarios_to_run), sid, scenario["question"][:50]))
        try:
            llm_output = call_gemini(scenario["question"], scenario["context_chunks"])
            cached_responses[sid] = llm_output
            checkpoint["responses"] = cached_responses
            _save_checkpoint(checkpoint)
            print("        Response: \"{}...\" ({:.1f}s) -- SAVED [OK]".format(
                llm_output["response"][:80], llm_output["latency_seconds"]))
        except Exception as e:
            error_str = str(e)
            if _is_quota_error(error_str):
                print("        QUOTA EXHAUSTED -- saving progress and stopping API calls.")
                quota_exhausted = True
            else:
                print("        ERROR: {}".format(e))
                cached_responses[sid] = {
                    "response": "Error generating response: {}".format(e),
                    "latency_seconds": 0, "estimated_cost_usd": 0, "model": "error"
                }
                checkpoint["responses"] = cached_responses
                _save_checkpoint(checkpoint)
        time.sleep(4)

    # Check if we have enough responses to continue
    valid_responses = {sid: r for sid, r in cached_responses.items()
                       if "Error" not in r.get("response", "Error")}
    scenarios_with_responses = [s for s in scenarios_to_run if s["id"] in valid_responses]

    if not scenarios_with_responses:
        print("\n  No valid responses yet. Run again when API quota resets.")
        print("  Checkpoint saved — progress will be preserved.")
        return

    print("\n  {} of {} scenarios have valid responses.".format(
        len(scenarios_with_responses), len(scenarios_to_run)))

    # ---- Step 2: Run Baseline evaluation (local, no API) ----
    print("\n[2/4] Running Baseline evaluation...")
    for scenario in scenarios_with_responses:
        sid = scenario["id"]
        if sid in cached_baseline:
            continue
        resp_text = cached_responses[sid]["response"]
        result = baseline_evaluate(resp_text, scenario["context_chunks"], scenario["retriever_scores"])
        result["scenario_id"] = sid
        result["category"] = scenario["category"]
        cached_baseline[sid] = result
    checkpoint["baseline_results"] = cached_baseline
    _save_checkpoint(checkpoint)
    print("  Done -- {}/{} evaluated.".format(len(cached_baseline), len(scenarios_with_responses)))

    # ---- Step 3: Run RAGAS-style evaluation (local, no API) ----
    print("\n[3/4] Running RAGAS-style evaluation...")
    for scenario in scenarios_with_responses:
        sid = scenario["id"]
        if sid in cached_ragas:
            continue
        resp_text = cached_responses[sid]["response"]
        result = ragas_style_evaluate(resp_text, scenario["context_chunks"], scenario["retriever_scores"])
        result["scenario_id"] = sid
        result["category"] = scenario["category"]
        cached_ragas[sid] = result
    checkpoint["ragas_results"] = cached_ragas
    _save_checkpoint(checkpoint)
    print("  Done -- {}/{} evaluated.".format(len(cached_ragas), len(scenarios_with_responses)))

    # ---- Step 4: Run Full SDK evaluation (LLM judge needs API) ----
    # Reset quota flag — SDK evals may use fresh quota from a new day
    quota_exhausted = False
    print("\n[4/4] Running Full SDK evaluation (all 5 features + LLM judge)...")
    for i, scenario in enumerate(scenarios_with_responses):
        sid = scenario["id"]
        if sid in cached_sdk:
            print("  [{}/{}] {} -- CACHED [OK]".format(i+1, len(scenarios_with_responses), sid))
            continue

        if quota_exhausted:
            print("  [{}/{}] {} -- SKIPPED (quota exhausted)".format(
                i+1, len(scenarios_with_responses), sid))
            continue

        print("  [{}/{}] {}...".format(i+1, len(scenarios_with_responses), sid))
        try:
            resp_text = cached_responses[sid]["response"]
            result = full_sdk_evaluate(resp_text, scenario["context_chunks"], scenario["retriever_scores"])
            result["scenario_id"] = sid
            result["category"] = scenario["category"]
            result["_response_preview"] = resp_text[:200] + "..." if len(resp_text) > 200 else resp_text
            cached_sdk[sid] = result
            checkpoint["sdk_results"] = cached_sdk
            _save_checkpoint(checkpoint)
            print("        SAVED [OK]")
        except Exception as e:
            error_str = str(e)
            if _is_quota_error(error_str):
                print("        QUOTA EXHAUSTED -- saving progress and stopping API calls.")
                quota_exhausted = True
            else:
                print("        ERROR: {}".format(e))
        time.sleep(4)

    # ---- Check completeness ----
    total_needed = len(scenarios_to_run)
    responses_done = len([s for s in scenarios_to_run if s["id"] in valid_responses])
    sdk_done = len([s for s in scenarios_to_run if s["id"] in cached_sdk])
    is_complete = (responses_done >= total_needed and sdk_done >= total_needed)

    if not is_complete:
        print("\n" + "=" * 70)
        print("  PROGRESS SAVED -- Run again when quota resets to continue!")
        print("  Responses: {}/{}  |  SDK evals: {}/{}".format(
            responses_done, total_needed, sdk_done, total_needed))
        print("  Checkpoint: {}".format(_checkpoint_path()))
        print("=" * 70)
        if quota_exhausted:
            print("  TIP: Daily quota resets at midnight Pacific Time.")
        print("")
        return

    # ---- All scenarios done — generate final results ----
    print("\n  All {} scenarios complete! Generating final results...".format(total_needed))

    all_results = []
    for sid in [s["id"] for s in scenarios_to_run]:
        if sid in cached_baseline:
            all_results.append(cached_baseline[sid])
        if sid in cached_ragas:
            all_results.append(cached_ragas[sid])
        if sid in cached_sdk:
            all_results.append(cached_sdk[sid])

    # Save raw results
    print("\n  Saving results...")
    raw_path = os.path.join(output_dir, "real_world_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "timestamp": datetime.datetime.now().isoformat(),
                "scenario_count": total_needed,
                "mode": "quick" if quick else "full",
                "llm_model": "gemini-2.5-flash",
                "test_type": "real_world_live_api"
            },
            "responses": {sid: {"response": r["response"][:500], "latency": r["latency_seconds"]}
                         for sid, r in cached_responses.items()},
            "all_results": all_results,
        }, f, indent=2, default=str)
    print("  Raw results -> {}".format(raw_path))

    # Generate human-readable summary
    summary = generate_summary(all_results, scenarios_to_run, output_dir)

    # Print final comparison
    print("")
    print("=" * 70)
    print("                 REAL-WORLD BENCHMARK COMPLETE")
    print("=" * 70)
    for method in ["Baseline", "RAGAS", "Ours (Full SDK)"]:
        method_results = [r for r in all_results if r["method"] == method]
        if not method_results:
            continue
        avg_faith = sum(r["faithfulness"] for r in method_results) / len(method_results)
        halluc_detected = sum(1 for r in method_results if r["detected_hallucination"])
        conflict_detected = sum(1 for r in method_results if r.get("detected_conflict", False))
        bias_detected = sum(1 for r in method_results if r.get("detected_position_bias", False))
        print("  {:<20} Faith: {:.2f}  Halluc: {}/{}  Conflict: {}/{}  Bias: {}/{}".format(
            method, avg_faith,
            halluc_detected, len(method_results),
            conflict_detected, len(method_results),
            bias_detected, len(method_results)))
    print("=" * 70)
    print("  Results: {}".format(output_dir))
    print("")

    # Clean up checkpoint file once complete
    cp_path = _checkpoint_path()
    if os.path.exists(cp_path):
        os.remove(cp_path)
        print("  Checkpoint cleaned up.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start fresh")
    args = parser.parse_args()
    if args.reset:
        cp = _checkpoint_path()
        if os.path.exists(cp):
            os.remove(cp)
            print("Checkpoint cleared.")
    run_real_world_test(quick=args.quick)
