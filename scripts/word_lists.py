# NLTK Word Lists
from nltk.corpus import stopwords

stop = stopwords.words("english")

# Custom Word Lists
listofknown_medications = [
    "clonidine",
    "quetiapine",
    "risperidone",
    "vyvanse",
    "adderall",
    "dexedrine",
    "wellbutrin",
    "focalinxr",
    "modafanil",
    "fluvoxamine",
    "serzone",
    "fluvoxamine",
    "prozac",
    "lexapro",
    "paxil",
    "celexa",
    "effexor",
    "zoloft",
    "cymbalta",
    "luvox",
    "pristiq",
    "remeron",
    "venlafaxine",
    "sarafem",
    "anafranil",
    "nortriptyline",
    "tofranil",
    "xanax",
    "klonopin",
    "ativan",
    "valium",
    "buspirone",
    "oxazepam",
    "aripiprazole",
    "dextroamphetamine",
    "ssrisnri",
    "clonazepam",
    "lorazepam",
    "temazepam",
    "alprazolam",
    "chlordiazepoxide",
    "flurazepam",
    "oxazepam",
    "triazolam",
    "divalproexsodium",
    "dronabinol",
    "nabilone",
    "duloxetine",
    "Atorvastatin",
    "Levothyroxine",
    "Metformin",
    "Lisinopril",
    "Amlodipine",
    "Metoprolol",
    "Albuterol",
    "Omeprazole",
    "Losartan",
    "Gabapentin",
    "Hydrochlorothiazide",
    "Sertraline",
    "Simvastatin",
    "Montelukast",
    "Escitalopram",
    "Acetaminophen; Hydrocodone",
    "Rosuvastatin",
    "Bupropion",
    "Furosemide",
    "Pantoprazole",
    "Trazodone",
    "Dextroamphetamine; Dextroamphetamine Saccharate; Amphetamine; Amphetamine Aspartate",
    "Fluticasone",
    "Tamsulosin",
    "Fluoxetine",
    "Carvedilol",
    "Duloxetine",
    "Meloxicam",
    "Clopidogrel",
    "Prednisone",
    "Citalopram",
    "Insulin Glargine",
    "Potassium Chloride",
    "Pravastatin",
    "Tramadol",
    "Aspirin",
    "Alprazolam",
    "Ibuprofen",
    "Cyclobenzaprine",
    "Amoxicillin",
    "Methylphenidate",
    "Allopurinol",
    "Venlafaxine",
    "Clonazepam",
    "Ethinyl Estradiol; Norethindrone",
    "Ergocalciferol",
    "Zolpidem",
    "Apixaban",
    "Glipizide",
    "Hydrochlorothiazide; Lisinopril",
    "Spironolactone",
    "Cetirizine",
    "Atenolol",
    "Oxycodone",
    "Buspirone",
    "Fluticasone; Salmeterol",
    "Topiramate",
    "Warfarin",
    "Estradiol",
    "Cholecalciferol",
    "Budesonide; Formoterol",
    "Lamotrigine",
    "Ethinyl Estradiol; Norgestimate",
    "Quetiapine",
    "Lorazepam",
    "Famotidine",
    "Folic Acid",
    "Azithromycin",
    "Acetaminophen; Oxycodone",
    "Hydroxyzine",
    "Insulin Lispro",
    "Diclofenac",
    "Loratadine",
    "Sitagliptin",
    "Clonidine",
    "Diltiazem",
    "Latanoprost",
    "Pregabalin",
    "Doxycycline",
    "Insulin Aspart",
    "Amitriptyline",
    "Paroxetine",
    "Ondansetron",
    "Tizanidine",
    "Lisdexamfetamine",
    "Rivaroxaban",
    "Glimepiride",
    "Propranolol",
    "Aripiprazole",
    "Finasteride",
    "Naproxen",
    "Levetiracetam",
    "Hydrochlorothiazide; Losartan",
    "Alendronate",
    "Fenofibrate",
    "Dulaglutide",
    "Oxybutynin",
    "Celecoxib",
    "Lovastatin",
    "Ezetimibe",
    "Cephalexin",
    "Empagliflozin",
    "Hydralazine",
    "Mirtazapine",
    "Cyanocobalamin",
    "Triamcinolone",
    "Amoxicillin; Clavulanate",
    "Baclofen",
    "Valproate",
    "Tiotropium",
    "Sumatriptan",
    "Donepezil",
    "Methotrexate",
    "Isosorbide",
    "Fluticasone; Vilanterol",
    "Ferrous Sulfate",
    "Thyroid",
    "Acetaminophen",
    "Valacyclovir",
    "Desogestrel; Ethinyl Estradiol",
    "Sulfamethoxazole; Trimethoprim",
    "Esomeprazole",
    "Valsartan",
    "Insulin Detemir",
    "Clindamycin",
    "Hydroxychloroquine",
    "Methocarbamol",
    "Diazepam",
    "Semaglutide",
    "Dexmethylphenidate",
    "Hydrochlorothiazide; Triamterene",
    "Ciprofloxacin",
    "Chlorthalidone",
    "Rizatriptan",
    "Nifedipine",
    "Insulin Degludec",
    "Norethindrone",
    "Risperidone",
    "Olmesartan",
    "Morphine",
    "Benazepril",
    "Meclizine",
    "Timolol",
    "Oxcarbazepine",
    "Drospirenone; Ethinyl Estradiol",
    "Liraglutide",
    "Dicyclomine",
    "Irbesartan",
    "Hydrocortisone",
    "Albuterol; Ipratropium",
    "Verapamil",
    "Memantine",
    "Prednisolone",
    "Metformin; Sitagliptin",
    "Nortriptyline",
    "Ropinirole",
    "Benzonatate",
    "Progesterone",
    "Ethinyl Estradiol; Levonorgestrel",
    "Mirabegron",
    "Methylprednisolone",
    "Acyclovir",
    "Docusate",
    "Olanzapine",
    "Nitroglycerin",
    "Bimatoprost",
    "Nitrofurantoin",
    "Pioglitazone",
    "Amlodipine; Benazepril",
    "Ketoconazole",
    "Clobetasol",
    "Testosterone",
    "Azelastine",
    "Fluconazole",
    "Brimonidine",
    "Desvenlafaxine",
    "Ranitidine",
    "Oseltamivir",
    "Levocetirizine",
    "Anastrozole",
    "Phentermine",
    "Sucralfate",
    "Sildenafil",
    "Mesalamine",
    "Carbamazepine",
    "Buprenorphine",
    "Acetaminophen; Codeine",
    "Flecainide",
    "Gemfibrozil",
    "Prazosin",
    "Lansoprazole",
    "Diphenhydramine",
    "Pramipexole",
    "Ethinyl Estradiol; Etonogestrel",
    "Dorzolamide; Timolol",
    "Ramipril",
    "Lithium",
    "Amiodarone",
    "Omega-3-acid Ethyl Esters",
    "Glyburide",
    "Acetaminophen; Butalbital; Caffeine",
    "Magnesium Salts",
    "Mupirocin",
    "Calcium",
    "Adalimumab",
    "Methimazole",
    "Budesonide",
    "Promethazine",
    "Doxazosin",
    "Labetalol",
    "Terazosin",
    "Cyclosporine",
    "Torsemide",
    "Medroxyprogesterone",
    "Calcium; Vitamin D",
    "Dorzolamide",
    "Dapagliflozin",
    "Liothyronine",
    "Sacubitril; Valsartan",
    "Beclomethasone",
    "Insulin Isophane",
    "Metronidazole",
    "Temazepam",
    "Fluticasone; Umeclidinium; Vilanterol",
    "Erythromycin",
    "Polyethylene Glycol 3350",
    "Nystatin",
    "Cefdinir",
    "Benztropine",
    "Tretinoin",
    "Mometasone",
    "Eszopiclone",
    "Betamethasone",
    "Erenumab",
    "Hydrochlorothiazide; Valsartan",
    "Minocycline",
    "Digoxin",
    "Empagliflozin; Metformin",
    "Nebivolol",
    "Levofloxacin",
    "Colchicine",
    "Ofloxacin",
    "Vortioxetine",
    "Linaclotide",
    "Umeclidinium",
    "Insulin Human; Insulin Isophane Human",
    "Ticagrelor",
    "Telmisartan",
    "Ketorolac",
    "Hydromorphone",
    "Epinephrine",
    "Doxepin",
    "Quinapril",
    "Umeclidinium; Vilanterol",
    "Fexofenadine",
    "Brimonidine; Timolol",
    "Letrozole",
    "Ranolazine",
    "Lurasidone",
    "Phenytoin",
    "Tadalafil",
    "Pancrelipase Amylase; Pancrelipase Lipase; Pancrelipase Protease",
    "Dexlansoprazole",
    "Isotretinoin",
    "Sodium Fluoride",
    "Solifenacin",
    "Bisoprolol",
    "Olopatadine",
    "Primidone",
    "Bumetanide",
    "Tolterodine",
    "Dexamethasone",
    "Chlorhexidine",
    "Sodium Salts",
    "Varenicline",
    "Zonisamide",
    "Calcitriol",
    "Emtricitabine; Tenofovir Disoproxil",
    "Terbinafine",
    "Fluocinonide",
    "Hydrochlorothiazide; Olmesartan",
    "Ziprasidone",
    "Estrogens, Conjugated",
    "Sulfasalazine",
    "Icosapent Ethyl",
    "Dexamethasone; Moxifloxacin",
    "Atomoxetine",
    "Formoterol; Mometasone",
    "Ketotifen",
    "Bisoprolol; Hydrochlorothiazide",
    "Sennosides",
    "Raloxifene",
    "Linagliptin",
    "Canagliflozin",
    "Alogliptin",
    "Sotalol",
    "Potassium Citrate",
    "Melatonin",
    "Isosorbide Dinitrate",
    "Guanfacine",
]


listofknown_medications = [
    x.lower() for x in listofknown_medications
]  # make all the words lowercase

conditions = ["autism", "ocd"]  # list of conditions to search for

slang_dict = {
    " tho ": " though ",
    "thru": " through ",
    "thx": " thanks ",
    " u ": " you ",
    " ur ": " your ",
    " yr ": " your ",
    " yrs ": " years ",
    " b ": " be ",
    " r ": " are ",
    " ppl ": " people ",
    " tmi ": " too much information ",
    " idk ": " i do not know ",
    " idc ": " i do not care ",
    " id ": " i would ",
    " imo ": " in my opinion ",
    " imho ": " in my humble opinion ",
    " tbh ": " to be honest ",
    " tbf ": " to be fair ",
    " tb ": " text back ",
    " bc ": " because ",
    " b/c ": " because ",
    " cuz ": " because ",
    " b4 ": " before ",
}

biasing_terms = ["ocd", "autism"]

# sources

# https://clincalc.com/DrugStats/Top300Drugs.aspx - list of top 300 drugs
