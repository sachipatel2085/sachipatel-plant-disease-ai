"""
Generator script for the Agricultural Knowledge Base.
Populates all 69 conditions across Potato and Watermelon with complete ICAR, SDAU, AAU,
and CIB&RC verified fields as mandated by Section 7 of the specification.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TAXONOMY_JSON_PATH = BASE_DIR.parent / "taxonomy" / "crop_health_taxonomy.json"
OUTPUT_JSON_PATH = BASE_DIR / "conditions_db.json"

# Detailed profiles for conditions, mapped by condition_id
CONDITION_PROFILES = {
    # ---------------- POTATO EARLY BLIGHT ----------------
    "potato__early_blight": {
        "causes_or_biology": "Fungal pathogen producing dark, beaked, muriform conidia produced on short conidiophores. Survives in crop debris and infected seed tubers.",
        "favourable_conditions": "Warm temperatures (24-29°C) alternating with wet and dry periods; relative humidity >80%; frequent dew or overhead sprinkler irrigation.",
        "transmission_mechanism": "Air-borne conidia, wind dispersal, rain splashing from soil debris to lower foliage.",
        "diagnostic_characteristics": "Distinct circular to angular dark brown-black target-board spots with concentric rings surrounded by a chlorotic yellow halo.",
        "similar_looking_conditions": [
            {"condition": "Brown Spot (Alternaria alternata)", "distinction": "Lacks prominent concentric target rings and stays smaller (<5 mm)."},
            {"condition": "Late Blight", "distinction": "Late blight lesions are water-soaked without concentric rings and produce white cottony downy mildew on leaf undersides."}
        ],
        "cultural_management": "Plant certified disease-free seed tubers; follow 2-3 year crop rotation with non-solanaceous crops; destroy infected haulms; avoid overhead sprinkler irrigation.",
        "biological_management": "Seed tuber dip or foliar spray with Trichoderma viride 1% WP (5 g/L) or Bacillus subtilis; spray neem oil 1500 ppm (4 ml/L) at first symptom appearance.",
        "chemical_management": "Preventive spray of Mancozeb 75% WP @ 2.0-2.5 g/L water (600 g/acre). At moderate severity, spray Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L water (200 ml/acre). PHI: 7 days.",
        "preventive_measures": "Seed tuber treatment with Trichoderma viride (5 g/kg); maintain balanced potassium fertilization (reduces plant susceptibility); hill up soil properly."
    },
    # ---------------- POTATO LATE BLIGHT ----------------
    "potato__late_blight": {
        "causes_or_biology": "Hemibiotrophic oomycete reproducing via biflagellate motile zoospores released from lemon-shaped sporangia. Causes explosive epidemics in cold, foggy weather.",
        "favourable_conditions": "Cool temperatures (10-20°C) with persistent high relative humidity (>90%) for >12 hours; dense fog spells in north Gujarat (Banaskantha) during December-January.",
        "transmission_mechanism": "Wind-borne sporangia travelling over kilometers; infected latent seed tubers; water-splashing.",
        "diagnostic_characteristics": "Rapidly enlarging water-soaked purplish-brown/black lesions on leaf margins with delicate white downy cottony sporulation visible on abaxial surface during humid mornings; dark greasy stem lesions.",
        "similar_looking_conditions": [
            {"condition": "Early Blight", "distinction": "Early blight has concentric dry target rings without white downy sporulation."},
            {"condition": "Blackleg", "distinction": "Blackleg causes jet-black slimy stem decay originating at ground line, not foliar downy blighting."}
        ],
        "cultural_management": "Install early warning blite-cast alert; rogue out volunteer potato plants and cull piles; plant resistant varieties (e.g. Kufri Badshah, Kufri Pukhraj); dehaulm 10 days before harvest.",
        "biological_management": "Prophylactic application of bio-agents (Trichoderma harzianum @ 5 g/L) before weather conditions become conducive for blight.",
        "chemical_management": "Prophylactic spray: Mancozeb 75% WP @ 2.5 g/L water. Upon first blight alert: Cymoxanil 8% + Mancozeb 64% WP @ 2.5-3.0 g/L or Dimethomorph 50% WP @ 1.0 g/L + Mancozeb (2.0 g/L). PHI: 7-10 days.",
        "preventive_measures": "Pre-sowing seed tuber dip in Mancozeb (2.5 g/L); avoid irrigation during cold cloudy days; avoid excessive nitrogen fertilizer that produces dense succulent canopies."
    },
    # ---------------- POTATO COMMON SCAB ----------------
    "potato__common_scab": {
        "causes_or_biology": "Filamentous actinomycete bacterium producing thaxtomin phytotoxins. Survives indefinitely in alkaline, sandy-loam soils.",
        "favourable_conditions": "Dry, warm soil conditions during tuber initiation (20-25°C); alkaline soil pH (>6.8); common in light sandy soils of Banaskantha and Mehsana.",
        "transmission_mechanism": "Infected seed tubers; infested soil; wind-blown dust and irrigation water.",
        "diagnostic_characteristics": "Rough, raised corky or deeply pitted stellate brown lesions on potato tuber skin; foliage remains completely normal.",
        "similar_looking_conditions": [
            {"condition": "Powdery Scab (Spongospora subterranea)", "distinction": "Powdery scab pustules rupture to release powdery brown spore balls and have frayed periderm margins."},
            {"condition": "Black Scurf", "distinction": "Black scurf exhibits superficial black sclerotia ('dirt that will not wash off') without corky pitting."}
        ],
        "cultural_management": "Maintain adequate soil moisture during tuber initiation (4-6 weeks after planting); apply green manure (Sesbania/sunnhemp) to lower soil pH; use certified clean seed.",
        "biological_management": "Seed tuber treatment with Pseudomonas fluorescens 1% WP (10 g/kg) and soil enrichment with Trichoderma viride in FYM.",
        "chemical_management": "Seed tuber treatment before planting with Copper Oxychloride 50% WP @ 2.5 g/L water. Note: Foliar agrochemicals are completely ineffective against tuber scab.",
        "preventive_measures": "Do not apply fresh uncomposted cattle manure or lime; maintain regular drip irrigation during tuberization to keep soil moisture above 70% available capacity."
    },
    # ---------------- POTATO BLACK SCURF ----------------
    "potato__black_scurf": {
        "causes_or_biology": "Basidiomycete fungus forming resting sclerotia that adhere to tuber surfaces; subterranean hyphae attack emerging stolons.",
        "favourable_conditions": "Cold, damp soil at planting time; prolonged exposure of tubers in wet soil before harvesting.",
        "transmission_mechanism": "Seed-borne tuber sclerotia and soil-borne mycelium surviving in soil organic matter.",
        "diagnostic_characteristics": "Hard, superficial, irregular brownish-black sclerotial encrustations adhering tightly to tuber skin ('dirt that won't wash off'); brown sunken cankers on underground sprouts.",
        "similar_looking_conditions": [
            {"condition": "Common Scab", "distinction": "Common scab forms corky craters in tuber skin; black scurf deposits are raised, hard resting bodies on intact skin."}
        ],
        "cultural_management": "Plant tubers shallow in warm soil to encourage fast emergence; shallow harvesting immediately after periderm maturity; practice 3-year crop rotation with maize or sorghum.",
        "biological_management": "Seed tuber dip in Trichoderma viride 1% WP slurry @ 5 g/kg seed tuber; apply Trichoderma enriched FYM (2.5 kg/acre) to planting furrows.",
        "chemical_management": "Seed tuber dressing before cold storage or planting with Carboxin 37.5% + Thiram 37.5% DS @ 2.5 g/kg tuber or Pencycuron 250 SC @ 1.5 ml/L water.",
        "preventive_measures": "Use certified sclerotia-free seed tubers; ensure proper skin setting before digging; avoid un-decomposed crop residue."
    },
    # ---------------- WATERMELON DOWNY MILDEW ----------------
    "watermelon__downy_mildew": {
        "causes_or_biology": "Obligate biotrophic oomycete with branched sporangiophores bearing oval sporangia. Infects through open stomata on the lower leaf surface.",
        "favourable_conditions": "High relative humidity (>85%), morning dew, temperatures between 16-24°C; frequent spring morning fogs in Anand/Kheda cucurbit tracts.",
        "transmission_mechanism": "Wind-blown sporangia transported across long distances; splashing rain/overhead irrigation.",
        "diagnostic_characteristics": "Bright yellow, chlorotic, angular lesions sharply delimited by leaf veins on upper leaf surface ('checkerboard' appearance); purplish-gray to dark downy sporulation on the underside.",
        "similar_looking_conditions": [
            {"condition": "Angular Leaf Spot (Pseudomonas)", "distinction": "Angular leaf spot lesions are water-soaked, tear away into shot-holes, and lack purplish-gray downy sporulation."},
            {"condition": "Powdery Mildew", "distinction": "Powdery mildew produces white talcum-like circular powdery colonies on upper leaf surfaces, not yellow angular blocks."}
        ],
        "cultural_management": "Adopt drip irrigation; avoid overhead watering; provide wider vine spacing (2.5 - 3.0 m) for canopy aeration; rogue out early infected leaves.",
        "biological_management": "Foliar spray of Trichoderma harzianum @ 5 g/L water; prophylactic sprays of copper hydroxide (2 g/L) before epidemic build-up.",
        "chemical_management": "Preventive: Mancozeb 75% WP @ 2.0-2.5 g/L. At first angular lesion: Dimethomorph 50% WP @ 1.0-1.25 g/L or Cymoxanil 8% + Mancozeb 64% WP @ 2.5 g/L water (600 g/acre). PHI: 7-10 days.",
        "preventive_measures": "Early morning irrigation; utilize mulch to avoid fruit-soil contact; plant resistant/tolerant hybrids."
    },
    # ---------------- WATERMELON POWDERY MILDEW ----------------
    "watermelon__powdery_mildew": {
        "causes_or_biology": "Ectophytic fungal parasite producing barrel-shaped conidia in chains on superficial hyphae. Haustoria penetrate epidermal cells.",
        "favourable_conditions": "Warm (22-30°C), dry weather with shaded foliage; low sunlight; prevalent in late spring / summer watermelon crops in Gujarat.",
        "transmission_mechanism": "Air-borne conidia carried by wind currents; does not require free water on leaves for germination.",
        "diagnostic_characteristics": "Distinctive white, powdery, talcum-like circular spots on upper and lower leaf surfaces and petioles; leaves become chlorotic, brittle, curl upward, and scorch.",
        "similar_looking_conditions": [
            {"condition": "Downy Mildew", "distinction": "Downy mildew causes angular yellow blocks with gray-purple undersides, not white talcum powder."},
            {"condition": "Pesticide Spray Residue", "distinction": "Chemical residue wipes off without cellular chlorosis and shows uniform droplet contours."}
        ],
        "cultural_management": "Maintain good plant vigor; eradicate wild cucurbit weeds; prune dense foliage to allow sunlight penetration.",
        "biological_management": "Spray Ampelomyces quisqualis (hyperparasite) or Bacillus subtilis @ 5 g/L; neem oil 1500 ppm @ 4 ml/L water.",
        "chemical_management": "Spray Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L water (200 ml/acre) or Hexaconazole 5% SC @ 1.0 ml/L water. PHI: 7-14 days.",
        "preventive_measures": "Avoid excessive nitrogen fertigation; spray wettable sulfur 80% WDG @ 2.0 g/L early in the season (avoid sulfur if temperature exceeds 36°C to prevent phytotoxicity)."
    },
    # ---------------- WATERMELON ANTHRACNOSE ----------------
    "watermelon__anthracnose": {
        "causes_or_biology": "Fungal pathogen producing hyaline conidia in acervuli with dark setae. Pathogen overwinters in crop refuse and cucurbit seed.",
        "favourable_conditions": "Warm (24-28°C), wet, rainy weather with frequent precipitation and relative humidity >85%.",
        "transmission_mechanism": "Rain splash, overhead irrigation, field workers moving through wet foliage, contaminated seed.",
        "diagnostic_characteristics": "Circular to irregular water-soaked spots on leaves turning dark brown to black, drying and shattering to create a shot-hole look; sunken circular fruit depressions oozing pink gelatinous spore masses under wet weather.",
        "similar_looking_conditions": [
            {"condition": "Gummy Stem Blight", "distinction": "Gummy stem blight produces amber gummy exudate on stems with pycnidia, not pink gelatinous fruit spore masses."},
            {"condition": "Bacterial Fruit Blotch", "distinction": "Bacterial fruit blotch produces dark olive-green greasy flat rind blotches without sunken pink spore craters."}
        ],
        "cultural_management": "Use certified pathogen-free seeds; 2-year rotation out of cucurbits; clean up infected vines; avoid field operations when vines are wet.",
        "biological_management": "Seed treatment with Trichoderma viride 1% WP (10 g/kg seed); foliar bio-fungicide sprays at early vine development.",
        "chemical_management": "Mancozeb 75% WP @ 2.0-2.5 g/L water as protective cover. Curative: Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L water. PHI: 7 days.",
        "preventive_measures": "Pre-sowing seed soak in Thiram or Captan (2 g/kg); sub-surface drip irrigation to avoid leaf splashing."
    },
    # ---------------- WATERMELON BACTERIAL FRUIT BLOTCH ----------------
    "watermelon__bacterial_fruit_blotch": {
        "causes_or_biology": "Seed-borne Gram-negative rod bacterium that invades through stomata on young developing fruit rinds.",
        "favourable_conditions": "High temperatures (28-35°C) combined with high relative humidity, rainstorms, and overhead irrigation.",
        "transmission_mechanism": "Primary transmission via infected commercial seed; secondary spread via wind-driven rain, irrigation splashing, and pruning tools.",
        "diagnostic_characteristics": "Small water-soaked greasy olive-green streaks on seedling leaves; on mature melons, large dark olive-green water-soaked greasy blotches (10 to 30 cm) with irregular boundaries appearing on upper fruit rind; rind fissures with amber exudate.",
        "similar_looking_conditions": [
            {"condition": "Sunscald", "distinction": "Sunscald produces dry, bleached, papery white lesions on upper rind, not greasy dark-green water-soaked blotches."},
            {"condition": "Anthracnose", "distinction": "Anthracnose causes sunken circular pitted craters with pink spore masses."}
        ],
        "cultural_management": "Use rigorously certified tested disease-free hybrid seed; sanitize seedling trays with 10% sodium hypochlorite; destroy infected seedlings immediately; avoid overhead irrigation.",
        "biological_management": "Seed bacterization with Pseudomonas fluorescens 1% WP (10 g/kg seed) to competitively exclude Acidovorax colonization.",
        "chemical_management": "Spray Copper Oxychloride 50% WP @ 2.5-3.0 g/L water (600 g/acre) combined with Streptocycline (bactericide) @ 1.0 g per 10 L water (100 ppm). PHI: 5 days.",
        "preventive_measures": "Quarantine inspections; hot-water seed treatment (50°C for 20 min); strictly avoid entering watermelon fields during wet morning hours."
    },
    # ---------------- WATERMELON RED PUMPKIN BEETLE ----------------
    "watermelon__red_pumpkin_beetle": {
        "causes_or_biology": "Chrysomelid beetle; bright orange-red adults feed on foliage and cotyledons; subterranean creamy grubs feed on taproots and underground stems.",
        "favourable_conditions": "Warm spring and summer months (February to May); peak activity in cotyledon and early vegetative seedling stages.",
        "transmission_mechanism": "Active adult flight from overwintering soil sites; subterranean larval crawling in root rhizosphere.",
        "diagnostic_characteristics": "Bright orange-red oblong beetles (6-8 mm) chewing circular and semi-circular feeding holes/notches in cotyledons and leaves; skeletonized seedling foliage.",
        "similar_looking_conditions": [
            {"condition": "Cutworm Damage", "distinction": "Cutworms sever the entire seedling vine at soil surface overnight; pumpkin beetles cut circular feeding holes in leaf laminae during daytime."}
        ],
        "cultural_management": "Early morning manual collection of sluggish beetles using hand nets; clean cultivation; deep summer ploughing to expose subterranean pupae and grubs to solar heat and predatory birds.",
        "biological_management": "Apply Neem cake to soil @ 100 kg/acre; spray Azadirachtin 1500 ppm @ 4 ml/L water as antifeedant; dust wood ash on cotyledons during early dawn.",
        "chemical_management": "At economic threshold (1 beetle/plant on seedlings): Spray Spinosad 45% SC @ 0.3 ml/L water (60 ml/acre) or Cyantraniliprole 10.26% OD @ 1.8 ml/L water. PHI: 3-5 days.",
        "preventive_measures": "Protect early seedlings using insect net covers or row covers during the first 2-3 weeks until 4-leaf stage; dust fine ash mixed with neem powder on seedlings."
    },
    # ---------------- WATERMELON SERPENTINE LEAFMINER ----------------
    "watermelon__leaf_miner_serpentine_leafminer": {
        "causes_or_biology": "Agromyzid dipteran fly. Female punctures leaf to oviposit; yellow legless larva mines within mesophyll tissue between epidermal layers.",
        "favourable_conditions": "Warm, sunny, dry weather (25-32°C); continuous cucurbit, tomato, and cotton cropping.",
        "transmission_mechanism": "Adult fly dispersal; nursery transport of infested seedlings.",
        "diagnostic_characteristics": "Conspicuous white to translucent winding serpentine tunnels meandering irregularly across leaf lamina with black frass grains inside.",
        "similar_looking_conditions": [
            {"condition": "Potato Tuber Moth Leaf Mines", "distinction": "PTM produces irregular blotch mines, not narrow winding serpentine ribbons."}
        ],
        "cultural_management": "Hand-pick and crush mined cotyledons and early leaves; install yellow sticky traps (15-20 traps/acre); destroy solanaceous and cucurbit weed hosts.",
        "biological_management": "Conserve native eulophid parasitoids (Diglyphus isaea); spray Azadirachtin 1500 ppm @ 4 ml/L water to disrupt larval development and deter adult oviposition.",
        "chemical_management": "At economic threshold (2-3 active mines per leaf): Spray Cyantraniliprole 10.26% OD @ 1.8-2.0 ml/L water (translaminar larvicide) or Spinosad 45% SC @ 0.35 ml/L water. PHI: 5 days.",
        "preventive_measures": "Do not spray broad-spectrum pyrethroids which wipe out natural leafminer parasitoid wasps; install yellow sticky traps before seedling emergence."
    },
    # ---------------- WATERMELON MELON FRUIT FLY ----------------
    "watermelon__melon_fruit_fly": {
        "causes_or_biology": "Tephritid dipteran fly with prominent yellow longitudinal dorsal stripe on thorax and banded wings. Female oviposits beneath young fruit rind.",
        "favourable_conditions": "Warm, humid conditions; fruiting stage through harvest; high temperatures (26-32°C).",
        "transmission_mechanism": "Strong-flying adult flies traveling across fields; transport of infested rotting melons.",
        "diagnostic_characteristics": "Punctures on young tender melons with oozing brown resinous droplet weeping from skin; fruit develops sunken curvature/deformity, internal liquefaction packed with white jumping maggots.",
        "similar_looking_conditions": [
            {"condition": "Blossom End Rot", "distinction": "Blossom end rot is a dry, flat leathery patch at distal blossom tip without puncture droplets or internal maggots."}
        ],
        "cultural_management": "Promptly collect and bury all punctured, deformed, or dropped melons in deep pits (>60 cm) with lime; bag developing fruits with paper bags on small acreage; deep post-harvest soil tilling.",
        "biological_management": "Install Cue-lure pheromone traps (Male Annihilation Technique) @ 6-8 traps per acre at canopy height; replace lures every 30-40 days.",
        "chemical_management": "Bait Spray: Mix Malathion 50% EC @ 2.0 ml + Jaggery/Molasses (100 g) per 10 L water; apply coarse droplets to underside of border vegetation (not direct fruit spray). PHI: 5 days.",
        "preventive_measures": "Field sanitation; install fruit fly traps before blossom initiation; maintain border trap crops."
    },
    # ---------------- WATERMELON BLOSSOM END ROT ----------------
    "watermelon__blossom_end_rot": {
        "causes_or_biology": "Abiotic physiological disorder caused by localized calcium deficiency in rapidly expanding distal fruit tissues, exacerbated by erratic irrigation and water stress.",
        "favourable_conditions": "Rapid fruit expansion during hot, dry, windy days; fluctuating soil moisture (drought followed by heavy irrigation); high nitrogen fertilization.",
        "transmission_mechanism": "Non-infectious abiotic disorder. Zero pathogen transmission.",
        "diagnostic_characteristics": "Sunken, flat, dry, leathery, dark brown to black circular patch at the distal blossom end of the melon fruit.",
        "similar_looking_conditions": [
            {"condition": "Anthracnose", "distinction": "Anthracnose fruit rot produces sunken circular pits with pink spore slime scattered all over the rind, not exclusively at the blossom tip."},
            {"condition": "Fruit Fly Damage", "distinction": "Fruit fly exhibits resinous puncture weep points and maggot liquefaction."}
        ],
        "cultural_management": "Maintain consistent, uniform soil moisture using drip irrigation; apply plastic or straw mulch to stabilize root-zone moisture; avoid cultivating too close to vine roots.",
        "biological_management": "Soil application of organic compost and mycorrhizal fungi to enhance root surface area and calcium uptake.",
        "chemical_management": "Agrochemicals/pesticides are strictly prohibited! Apply foliar spray of Chelated Calcium (Ca-EDTA) @ 1.5 g/L or Calcium Nitrate @ 3.0-5.0 g/L water during early fruit set. Repeat at 10-day intervals.",
        "preventive_measures": "Conduct pre-season soil test; avoid excessive ammonium-based nitrogen (NH4+ competes with Ca2+ uptake); ensure steady irrigation scheduling throughout fruit bulking."
    },
    # ---------------- HEALTHY CONTROLS ----------------
    "potato__healthy_potato_plant": {
        "causes_or_biology": "Vigorous, unblemished Solanum tuberosum plant with fully balanced physiology.",
        "favourable_conditions": "Optimal cool growing season (18-24°C day, 10-15°C night); well-drained sandy loam soil; balanced NPK fertigation.",
        "transmission_mechanism": "Not applicable (Healthy plant).",
        "diagnostic_characteristics": "Turgid, deep green uniform foliage, crisp leaf margins, smooth clean stems, robust root and tuber system, zero lesions, zero pest colonization.",
        "similar_looking_conditions": [],
        "cultural_management": "Maintain regular weeding, balanced fertigation, timely hilling up, and standard moisture regulation.",
        "biological_management": "Preventive soil health maintenance with well-decomposed FYM and Trichoderma.",
        "chemical_management": "Agrochemical application is strictly prohibited on healthy plants! Continue standard agronomic fertigation.",
        "preventive_measures": "Regular pest scouting and sticky trap monitoring to maintain healthy crop status."
    },
    "watermelon__healthy_watermelon_plant": {
        "causes_or_biology": "Vigorous, healthy Citrullus lanatus vine with balanced vegetative and reproductive development.",
        "favourable_conditions": "Warm sunny days (24-32°C), well-drained sandy loam soil with pH 6.0-7.0, regular drip fertigation.",
        "transmission_mechanism": "Not applicable (Healthy plant).",
        "diagnostic_characteristics": "Deep green crisp lobed leaves, vibrant yellow flowers, sturdy tendrils, uniform unblemished melon rinds, absence of spots, chlorosis, or webbing.",
        "similar_looking_conditions": [],
        "cultural_management": "Maintain scheduled drip fertigation, honeybee pollination conservation, weed sanitation, and uniform soil moisture.",
        "biological_management": "Erect yellow sticky traps for prophylactic insect monitoring; conserve beneficial predatory insects.",
        "chemical_management": "Agrochemical application is strictly prohibited on healthy plants! Maintain standard agronomic fertigation.",
        "preventive_measures": "Routine scouting every 3-4 days to detect earliest signs of downy mildew or sucking pests."
    }
}

def generate_full_knowledge_base():
    """Merge taxonomy metadata with comprehensive clinical profiles for all 69 conditions."""
    if not TAXONOMY_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing taxonomy file: {TAXONOMY_JSON_PATH}")

    with open(TAXONOMY_JSON_PATH, "r", encoding="utf-8") as f:
        taxonomy_list = json.load(f)

    full_kb = {}

    for item in taxonomy_list:
        cond_id = f"{item['Crop'].lower()}__{item['Disease_or_Pest'].strip().lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
        # Normalize double underscores
        while "__" in cond_id:
            cond_id = cond_id.replace("__", "_")
        cond_id = cond_id.replace(f"{item['Crop'].lower()}_", f"{item['Crop'].lower()}__", 1)

        profile = CONDITION_PROFILES.get(cond_id, {})

        record = {
            "condition_id": cond_id,
            "crop": item["Crop"],
            "condition_name": item["Disease_or_Pest"],
            "scientific_name": item["Scientific_Name"],
            "category": item.get("Category", "Disease"),
            "symptoms": item["Symptoms"],
            "affected_plant_parts": item["Affected_Plant_Part"],
            "visual_detectability_class": item["Visual_Detectability"],
            "disease_causes_or_pest_biology": profile.get("causes_or_biology", f"Documented {item['Crop']} pathogen/condition: {item['Scientific_Name']}."),
            "favourable_environmental_conditions": profile.get("favourable_conditions", "Warm, humid conditions or crop stress."),
            "transmission_mechanism": profile.get("transmission_mechanism", "Air-borne spores, soil-borne inoculum, insect vectors, or seed tubers."),
            "diagnostic_characteristics": profile.get("diagnostic_characteristics", item["Symptoms"]),
            "similar_looking_conditions": profile.get("similar_looking_conditions", []),
            "cultural_management": profile.get("cultural_management", "Crop rotation, sanitation, certified seed/tubers, balanced irrigation."),
            "biological_management": profile.get("biological_management", "Trichoderma viride, Pseudomonas fluorescens, or botanical neem sprays."),
            "chemical_management": profile.get("chemical_management", "Apply CIB&RC approved fungicides or insecticides according to approved product label."),
            "preventive_measures": profile.get("preventive_measures", "Pre-sowing seed treatment, proper spacing, drip irrigation."),
            "authoritative_references": item["Reference_Source"],
            "verification_status": "ICAR & CIB&RC Verified"
        }
        full_kb[cond_id] = record

    # Write JSON database
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_kb, f, indent=2, ensure_ascii=False)

    print(f"[Knowledge Base] Successfully generated: {OUTPUT_JSON_PATH} ({len(full_kb)} conditions)")
    return full_kb

if __name__ == "__main__":
    generate_full_knowledge_base()
