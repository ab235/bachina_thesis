from dataclasses import dataclass
from typing import Tuple

DEFAULT_REFERENCE = "rpi-student-handbook"
DEFAULT_THRESHOLD = 0.8


@dataclass(frozen=True)
class GroundTruthCase:
    id: str
    question: str
    answer: str
    reference: str
    pass_threshold: float = DEFAULT_THRESHOLD


GROUND_TRUTH: Tuple[GroundTruthCase, ...] = (
    GroundTruthCase(
        id="sbr-001",
        question="What is the stated purpose of the Student Bill of Rights in Article I?",
        answer="Its purpose is to set forth how students' fundamental rights as citizens are applied to student members of the Rensselaer community.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-002",
        question="According to Article II Section A, how does Rensselaer describe access to admission?",
        answer="The Institute states expectations relevant to success and is open to all students qualified by admission standards, barring discrimination on protected bases (and it also seeks socioeconomic diversity).",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-003",
        question="What does Article II Section B guarantee about facilities and services?",
        answer="Facilities and services normally available to students are open to all students without discrimination on protected bases; age/year cannot be used arbitrarily, though differential access may be allowed for valid educational/resource reasons, and the Institute will endeavor to secure equal access to public facilities in the local community.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-004",
        question="What does Article II Section C promise about financial aid information?",
        answer="Prospective students have a right to a written explanation of financial aid eligibility and continuation requirements, and aid recipients must receive an explanation for later changes in aid.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-005",
        question="How must professors approach classroom dialogue under Article III Section A?",
        answer="Professors must encourage free discussion, inquiry, and expression; students may take reasoned exception to course data/views and reserve judgment on matters of opinion.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-006",
        question="What course information must be provided at the start of a class?",
        answer="At the beginning of each course, students should receive (1) a syllabus with content/objectives, (2) evaluation policies and how the final grade is determined (including factors like homework/exams/projects/papers/labs/attendance, and reasons for changes), and (3) a statement defining academic dishonesty (with attention to collaboration).",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-007",
        question="How must coursework be evaluated according to Article III Section C?",
        answer="Student performance must be evaluated on an academic basis-not on opinions or unrelated conduct-and students must have protection through orderly procedures against prejudiced or capricious evaluation.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-008",
        question="What confidentiality expectations are placed on professors in Article III Section D?",
        answer="Information professors learn about student activities, views, beliefs, and political associations in their instructional/advising roles is confidential; those providing confidential recommendations with student permission must be honest and fair.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-009",
        question="Summarize the protections in Article IV about student records.",
        answer="The Institute must publish explicit policies on what is in the permanent educational record and disclosure conditions; academic/financial/disciplinary/medical records should be separate; transcripts contain only academic status and students/advisers may see them; records are available only to authorized persons or others with student permission; no records should reflect political activities/beliefs; inactive non-academic/nonfinancial records should be periodically destroyed; students may view/contest official records except admissions application and counseling/medical records.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-010",
        question="What does Article V Section A state about student organizations?",
        answer="Students may organize/join lawful associations; a student organization's policies/actions are determined by its membership within limits set by the Rensselaer Union and other appropriate Institute bodies, and affiliation with outside organizations alone does not disqualify recognition.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-011",
        question="What freedoms are given in Article V Section B?",
        answer="Students/groups may examine and discuss issues, express opinions publicly/privately, support causes by orderly means including peaceful assembly that does not disrupt Institute operations, and invite/hear any person of their choosing (with an obligation to comply with Institute procedures for preparation/security/appropriateness).",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-012",
        question="How does Article V Section C treat student media?",
        answer="The Union must provide sufficient editorial freedom to preserve integrity as vehicles for responsible free expression; it must clarify roles/standards/limits; editors/managers are protected from arbitrary discipline/removal for disapproved content and may be removed only for proper stated causes via orderly procedures; student media must state opinions are not necessarily those of the Institute or student body.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-013",
        question="What is promised in Article V Section D?",
        answer="Students may express views on institutional policy and matters of general student interest; the student body must have a means to provide input on institutional policy affecting academic and student affairs; student government's role/responsibilities must be explicit and reviewed through orderly prescribed procedures.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-014",
        question="What does Article VI Section A say about off-campus freedom?",
        answer="As citizens, students enjoy the same freedom of speech, peaceful assembly, and petition as other citizens; both off-campus and on-campus they are expected to act civilly, respectfully, and lawfully.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-015",
        question="How does Article VI Section B describe Institute involvement when laws are broken?",
        answer="When student activities appear to violate law, Institute offices will apprise students of sources of legal counsel/assistance; civil authorities may impose penalties, and the Institute will not duplicate public authority-but it reserves the right to address off-campus conduct that violates the Grounds for Disciplinary Action through its judicial process.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-016",
        question="Summarize Article VII Section A's procedural safeguards.",
        answer="Discipline must provide procedural fairness; regulations should be clear/specific and communicated; disciplinary action may be initiated for threats to safety, endangerment of property, or disruptive conduct; procedures consider circumstances and guarantee an individual inquiry/hearing; jurisdiction/official responsibilities/procedures (including appeal rights) must be available in advance and penalties imposed under prescribed procedures.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-017",
        question="What rights does Article VII Section B highlight?",
        answer="Students must be informed of charges and given a fair chance to refute them; actions cannot be arbitrary; there must be provisions for appeal; standards of conduct should be formulated with student input and published/communicated in advance except in extraordinary circumstances.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-018",
        question="What limits on searches are described in Article VII Section C?",
        answer="Student premises/property cannot be searched or seized without an externally issued warrant or comparable internal equivalent, or without the student's knowledge/approval, except where officials reasonably believe safety is involved; for premises not controlled by the Institute, ordinary lawful search requirements apply.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-019",
        question="How are students protected during investigations per Article VII Section D?",
        answer="Students detected/charged in serious violations must be informed of their rights; no harassment may be used to coerce admissions of guilt or information about conduct/others.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-020",
        question="Describe the interim suspension rules in Article VII Section E.",
        answer="The Director of Student Rights, Responsibilities and Conduct may impose interim suspension during investigation if the student's continued presence/participation could endanger security/health/safety or person/property; interim suspension denies campus access (classes, facilities, residence halls, activities/privileges), does not replace the normal conduct process, and the student may request written review to the Dean of Students, who reviews within five Institute business days.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-021",
        question="How does the Student Conduct section balance individuality and standards?",
        answer="It states Rensselaer does not seek to impose a common morality out of respect for individuality and privacy, but still has responsibility to establish standards of conduct within the campus community.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-022",
        question="When can off-campus misconduct lead to Institute discipline?",
        answer="Off-campus misconduct is not typically the basis for discipline except as prescribed in the Student Bill of Rights; however, if it threatens person/property within the Rensselaer community (or under other circumstances described), it can result in disciplinary review/action.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-023",
        question="What responsibility do recognized student organizations accept?",
        answer="Recognition means organizations accept corporate responsibility to protect community members/guests from rights violations in group activities and members' activities; officers may be required but that does not diminish corporate responsibility, and officers' failures in official capacity may also trigger individual discipline.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-024",
        question="Who administers student disciplinary and judicial processes?",
        answer="Authority is vested exclusively in the Dean of Students Office, by delegation from the President.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-025",
        question="What is the first listed Ground for Disciplinary Action?",
        answer="Conduct that could be construed as a violation of any federal, state, or local law.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-026",
        question="How does the handbook address disruptive conduct?",
        answer="It prohibits conduct that disrupts or interferes with others' personal/group rights or with Institute activities, including access to facilities and performance of normal duties.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-027",
        question="What behaviors violate the rule against unlawful intrusion or seizure?",
        answer="Violations include theft/possession of stolen property, unauthorized keys/access devices, unauthorized entry, and refusing to leave or release property when ordered by someone with jurisdiction.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-028",
        question="How is property damage treated?",
        answer="Damage to property, including vandalism, is a disciplinary ground.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-029",
        question="What is said about academic dishonesty in the Grounds for Disciplinary Action?",
        answer="Academic dishonesty (as defined in the handbook) is a ground for disciplinary action.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-030",
        question="How does the handbook define fraud as a violation?",
        answer="Fraud includes (but is not limited to) forgery, misuse, and/or alteration of Institute records, documents, or identification.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-031",
        question="What is the policy on controlled substances?",
        answer="Use, possession, or distribution of controlled substances (illegal drugs) and precursors/paraphernalia is prohibited, except as expressly permitted by law and Institute regulations; additionally, being somewhere for unlawful use/possession/distribution is also a violation.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-032",
        question="How is disorderly or indecent conduct addressed?",
        answer="Disorderly, lewd, harassing, or indecent conduct is a disciplinary ground.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-033",
        question="What about physical assaults or threats?",
        answer="Physical assault, or conduct that threatens/encourages/causes physical harm to persons or property, is a disciplinary ground.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-034",
        question="How is hazing treated in the handbook?",
        answer="Hazing (as defined in the handbook or by New York State) is a disciplinary ground.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-035",
        question="What expectations exist about cooperating with judicial proceedings?",
        answer="Willful failure/refusal to testify as a directed witness is a violation (with stated exceptions), and knowingly providing false testimony/evidence is also a violation.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-036",
        question="How does the handbook define conduct that endangers safety?",
        answer="Conduct endangering safety includes tampering with fire-warning/fire protection equipment, setting a fire on Institute property, using cooking equipment in unauthorized residence-hall areas, and reckless motor vehicle operation (among others).",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-037",
        question="What restrictions exist on weapons or dangerous materials?",
        answer="Use/possession/storage of dangerous weapons, chemicals, explosive devices, or materials (including firearms, air guns, prohibited knives, ammunition, fireworks, bombs, etc.) is a disciplinary ground.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-038",
        question="What is required when interacting with Institute officials?",
        answer="Students must comply with Institute officials performing duties, including providing valid ID and not knowingly furnishing false information.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-039",
        question="How is discriminatory harassment defined as a violation?",
        answer="It is engaging in oral/written/graphic/physical conduct that may harm, threaten, harass, demean, intimidate, or create a hostile environment for an individual/group based on a protected identity/category under the Institute's non-discrimination policy.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-040",
        question="Under what circumstances does the Institute assert jurisdiction over off-campus conduct based on who is affected?",
        answer="The Institute may assert jurisdiction over off-campus conduct when the behavior impacts a Rensselaer student, a Rensselaer-affiliated individual or group, or the Institute itself, even if the respondent was unaware of the victim’s affiliation.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-041",
        question="How can a student’s affiliation with Rensselaer establish Institute jurisdiction for off-campus behavior?",
        answer="Institute jurisdiction applies when a student leverages their status as a Rensselaer student or association with a Rensselaer organization in order to carry out or enable off-campus misconduct.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-042",
        question="When does participation in events connect off-campus behavior to Institute jurisdiction?",
        answer="Off-campus behavior falls under Institute jurisdiction when it occurs in connection with a Rensselaer-sponsored, Rensselaer-sanctioned, or recognized student organization event.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-043",
        question="What types of off-campus misconduct allow the Institute to intervene due to severity?",
        answer="The Institute may intervene in cases of off-campus misconduct that present a serious risk to health or safety, involve severe or dangerous conduct, or have the potential to significantly harm the Institute’s community or reputation.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-044",
        question="What standard of proof does the Rensselaer judicial system use?",
        answer="Preponderance of the evidence.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-045",
        question="What obligations do students have during a judicial inquiry?",
        answer="A student must be present when requested; they are responsible to answer fully and truthfully, with limited right to refuse specific questions only when answers would tend to incriminate them (and they must state reasons for refusal).",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-046",
        question="How can a student request a Judicial Board hearing?",
        answer="If the student does not accept the responsibility finding and sanctions, they may request a Judicial Board hearing in writing to the Board Chairperson via the Senior Judicial Administrator within three Institute business days of receiving the hearing officer's decision.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-047",
        question="What are the only grounds for appealing a judicial decision?",
        answer="Only demonstrated procedural error; material new evidence not discoverable earlier that would likely change the outcome; and/or sanctions that are not appropriate for the violations.",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-048",
        question="How does the academic integrity policy define academic fraud?",
        answer="Academic fraud is altering documentation relating to the grading process (e.g., changing exam solutions to negotiate a higher grade or tampering with an instructor's grade book).",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-049",
        question="What is the handbook's definition of plagiarism?",
        answer="Plagiarism is representing another's work or words as one's own by omitting acknowledgment or reference (including verbatim sentences without citation, adopting another's detailed argument as one's own, or using enhanced graphics without attribution).",
        reference=DEFAULT_REFERENCE,
    ),
    GroundTruthCase(
        id="sbr-050",
        question="What protection does the Good Samaritan Policy give to a person who receives medical help?",
        answer="An individual who receives emergency assistance/medical treatment due to alcohol or drug consumption-and completes the required assessment/education/treatment assigned through the Health Center-will not be subject to judicial action for violating the Institute Alcohol & Other Drugs Policy.",
        reference=DEFAULT_REFERENCE,
    ),
)

