from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union


@dataclass(frozen=True)
class KeywordCriterion:
    phrase: str
    weight: float = 1.0


KeywordInput = Union[str, Tuple[str, float], KeywordCriterion]


@dataclass(frozen=True)
class GroundTruthCase:
    """Question/answer pairs plus lightweight grading metadata."""

    id: str
    question: str
    answer: str
    keywords: Sequence[KeywordInput]
    reference: str
    pass_threshold: float = 1.0


KEYWORD_MAP: Dict[str, Tuple[str, ...]] = {
    "sbr-001": ("citizens", "rights", "Bill"),
    "sbr-002": ("admission", "qualified", "diverse"),
    "sbr-003": ("student", "discrimination", "equal"),
    "sbr-004": ("written", "funding", "right"),
    "sbr-005": ("discussion", "free", "course"),
    "sbr-006": ("syllabus", "grading", "dishonesty"),
    "sbr-007": ("academic", "judged", "evaluation"),
    "sbr-008": ("faculty", "confidential", "honest"),
    "sbr-009": ("records", "authorized", "contest"),
    "sbr-010": ("associations", "extramural", "discrimination"),
    "sbr-011": ("discuss", "speak", "invite"),
    "sbr-012": ("editorial", "freedom", "opinions"),
    "sbr-013": ("policy", "student government", "input"),
    "sbr-014": ("speech", "assembly", "citizens"),
    "sbr-015": ("legal counsel", "civil authorities", "safety"),
    "sbr-016": ("fair process", "rules", "appeal"),
    "sbr-017": ("charges", "refute", "appeal"),
    "sbr-018": ("search", "warrant", "safety"),
    "sbr-019": ("investigation", "harassment", "rights"),
    "sbr-020": ("interim suspension", "danger", "review"),
    "sbr-021": ("individuality", "standards", "rights"),
    "sbr-022": ("off-campus", "threatens community", "discipline"),
    "sbr-023": ("organizations", "officers", "responsibility"),
    "sbr-024": ("discipline", "Dean of Students", "authority"),
    "sbr-025": ("violate law", "disciplinary", "conduct"),
    "sbr-026": ("disrupts", "rights", "activities"),
    "sbr-027": ("theft", "unauthorized keys", "entry"),
    "sbr-028": ("damage", "property", "vandalism"),
    "sbr-029": ("academic dishonesty", "violation", "grounds"),
    "sbr-030": ("fraud", "forgery", "records"),
    "sbr-031": ("drugs", "possession", "distribution"),
    "sbr-032": ("disorderly", "lewd", "harassing"),
    "sbr-033": ("assault", "threatens", "harm"),
    "sbr-034": ("hazing", "handbook", "state"),
    "sbr-035": ("testify", "false", "proceedings"),
    "sbr-036": ("tampering", "fires", "reckless driving"),
    "sbr-037": ("weapons", "chemicals", "explosives"),
    "sbr-038": ("comply", "identification", "information"),
    "sbr-039": ("harms", "hostile", "protected"),
    "sbr-040": ("victim", "institute-related", "status"),
    "sbr-041": ("student status", "facilitate", "jurisdiction"),
    "sbr-042": ("sponsored events", "organization", "jurisdiction"),
    "sbr-043": ("infractions", "reputation", "safety"),
    "sbr-044": ("preponderance", "evidence", "standard"),
    "sbr-045": ("appear", "truthfully", "adviser"),
    "sbr-046": ("request", "three days", "defer sanctions"),
    "sbr-047": ("appeals", "new evidence", "sanctions"),
    "sbr-048": ("fraud", "documentation", "grading"),
    "sbr-049": ("plagiarism", "acknowledgment", "citing"),
    "sbr-050": ("assistance", "treatment", "judicial immunity"),
}


GROUND_TRUTH: List[GroundTruthCase] = [
    GroundTruthCase(
        id="sbr-001",
        question="What is the stated purpose of the Student Bill of Rights in Article I?",
        answer=(
            "The handbook affirms that students remain citizens whose fundamental rights cannot be obstructed, "
            "and the Student Bill of Rights exists to explain how those rights apply to members of the Rensselaer "
            "community."
        ),
        keywords=KEYWORD_MAP["sbr-001"],
        reference="student_handbook.txt:101-107",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-002",
        question="According to Article II Section A, how does Rensselaer describe access to admission?",
        answer=(
            "Rensselaer pledges to publish the expectations for student success and to keep admission open to all "
            "qualified applicants without discrimination based on protected characteristics while seeking students from "
            "diverse socioeconomic backgrounds."
        ),
        keywords=KEYWORD_MAP["sbr-002"],
        reference="student_handbook.txt:111-118",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-003",
        question="What does Article II Section B guarantee about facilities and services?",
        answer=(
            "Facilities and services normally available under Institute rules must be open to every student without "
            "discrimination, with age or class year restrictions used only for valid educational or resource reasons while "
            "the Institute works to secure equal access in the local community."
        ),
        keywords=KEYWORD_MAP["sbr-003"],
        reference="student_handbook.txt:121-127",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-004",
        question="What does Article II Section C promise about financial aid information?",
        answer=(
            "Prospective students have the right to a written explanation of financial aid eligibility and continuation "
            "requirements, and aid recipients must be told why their funding may change in later years."
        ),
        keywords=KEYWORD_MAP["sbr-004"],
        reference="student_handbook.txt:151-153",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-005",
        question="How must professors approach classroom dialogue under Article III Section A?",
        answer=(
            "Professors are expected to encourage free discussion, inquiry, and expression and to allow students to take "
            "reasoned exception to course content or opinions."
        ),
        keywords=KEYWORD_MAP["sbr-005"],
        reference="student_handbook.txt:155-157",
    ),
    GroundTruthCase(
        id="sbr-006",
        question="What course information must be provided at the start of a class?",
        answer=(
            "Article III Section B requires a syllabus with content and objectives, grading policies describing how work "
            "factors into the final grade, and a statement clarifying the definition of academic dishonesty, especially "
            "around collaboration."
        ),
        keywords=KEYWORD_MAP["sbr-006"],
        reference="student_handbook.txt:158-168",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-007",
        question="How must coursework be evaluated according to Article III Section C?",
        answer=(
            "Student performance must be judged solely on an academic basis, and students are protected through "
            "orderly procedures against prejudiced or capricious evaluation."
        ),
        keywords=KEYWORD_MAP["sbr-007"],
        reference="student_handbook.txt:169-171",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-008",
        question="What confidentiality expectations are placed on professors in Article III Section D?",
        answer=(
            "Information faculty learn about students’ activities, views, beliefs, or associations while teaching or advising "
            "must remain confidential, and those writing confidential statements must be honest and fair to both the "
            "recipient and the student."
        ),
        keywords=KEYWORD_MAP["sbr-008"],
        reference="student_handbook.txt:172-176",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-009",
        question="Summarize the protections in Article IV about student records.",
        answer=(
            "Rensselaer must publish what belongs in permanent records, keep academic, financial, disciplinary, and "
            "medical files separate, limit access to authorized persons, allow students and advisers to view transcripts, "
            "prohibit records of political beliefs, periodically destroy inactive non-academic files, and let students review "
            "and contest their official records except for admissions and counseling or medical documents."
        ),
        keywords=KEYWORD_MAP["sbr-009"],
        reference="student_handbook.txt:177-190",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-010",
        question="What does Article V Section A state about student organizations?",
        answer=(
            "Students are free to form lawful associations whose policies are set by their members, recognition cannot be "
            "denied solely for extramural affiliations, organizations may be asked for purpose statements and officer lists, "
            "must remain open to eligible students without discrimination, and access to facilities or resources cannot be "
            "withheld as censorship."
        ),
        keywords=KEYWORD_MAP["sbr-010"],
        reference="student_handbook.txt:191-205",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-011",
        question="What freedoms are given in Article V Section B?",
        answer=(
            "Students and groups may discuss any issues, speak publicly or privately, support causes through orderly, "
            "non-disruptive means, invite speakers of their choosing, and must follow Institute procedures for preparation "
            "and security while clarifying that sponsorship does not equal endorsement."
        ),
        keywords=KEYWORD_MAP["sbr-011"],
        reference="student_handbook.txt:206-215",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-012",
        question="How does Article V Section C treat student media?",
        answer=(
            "The Rensselaer Union must guarantee sufficient editorial freedom, clarify media roles and standards, protect "
            "editors from arbitrary removal, and require financed publications to note that opinions expressed are not "
            "necessarily those of the Institute or student body."
        ),
        keywords=KEYWORD_MAP["sbr-012"],
        reference="student_handbook.txt:216-225",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-013",
        question="What is promised in Article V Section D?",
        answer=(
            "Students may express views on institutional policy and issues of student interest, and student government roles "
            "and responsibilities must be explicit with formal, orderly procedures for reviewing its actions and providing "
            "input on academic and student affairs."
        ),
        keywords=KEYWORD_MAP["sbr-013"],
        reference="student_handbook.txt:226-230",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-014",
        question="What does Article VI Section A say about off-campus freedom?",
        answer=(
            "Students remain citizens entitled to freedoms of speech, peaceful assembly, and petition, and they are expected "
            "to conduct themselves civilly, respectfully, and lawfully both on and off campus."
        ),
        keywords=KEYWORD_MAP["sbr-014"],
        reference="student_handbook.txt:231-235",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-015",
        question="How does Article VI Section B describe Institute involvement when laws are broken?",
        answer=(
            "Student affairs staff will guide students to legal counsel, civil penalties remain the province of public authorities, "
            "the Institute will not duplicate civil proceedings but may address off-campus conduct that violates its rules, and "
            "a student's status will change only when their presence threatens campus safety."
        ),
        keywords=KEYWORD_MAP["sbr-015"],
        reference="student_handbook.txt:236-247",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-016",
        question="Summarize Article VII Section A's procedural safeguards.",
        answer=(
            "Discipline must be procedurally fair, rules must be clear and communicated, action targets conduct that "
            "threatens safety, property, or disrupts academics, each accused student gets an individual inquiry or hearing, "
            "and jurisdiction plus appeals must be defined with penalties imposed under prescribed procedures."
        ),
        keywords=KEYWORD_MAP["sbr-016"],
        reference="student_handbook.txt:248-258",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-017",
        question="What rights does Article VII Section B highlight?",
        answer=(
            "Students must be informed of the charges, given a fair chance to refute them, provided appeal options, and "
            "student leaders should help craft, publish, and communicate conduct standards in advance."
        ),
        keywords=KEYWORD_MAP["sbr-017"],
        reference="student_handbook.txt:259-264",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-018",
        question="What limits on searches are described in Article VII Section C?",
        answer=(
            "Student spaces or property cannot be searched or seized without an external warrant or comparable internal "
            "authorization or the student's knowledge and approval, unless safety concerns justify immediate action."
        ),
        keywords=KEYWORD_MAP["sbr-018"],
        reference="student_handbook.txt:265-269",
    ),
    GroundTruthCase(
        id="sbr-019",
        question="How are students protected during investigations per Article VII Section D?",
        answer=(
            "When serious violations are alleged, students must be informed of their rights and Institute officials are "
            "forbidden from using harassment to coerce admissions of guilt or accusations against others."
        ),
        keywords=KEYWORD_MAP["sbr-019"],
        reference="student_handbook.txt:270-272",
    ),
    GroundTruthCase(
        id="sbr-020",
        question="Describe the interim suspension rules in Article VII Section E.",
        answer=(
            "The Director of Student Rights, Responsibilities and Conduct may impose interim suspension when a student's "
            "presence endangers security, health, or safety; it removes access to classes, facilities, and activities, does not "
            "replace the normal conduct process, and the student can request a Dean of Students review within five "
            "Institute business days."
        ),
        keywords=KEYWORD_MAP["sbr-020"],
        reference="student_handbook.txt:273-283",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-021",
        question="How does the Student Conduct section balance individuality and standards?",
        answer=(
            "Rensselaer respects individuality and does not impose a common morality, yet it reminds students they must "
            "live within laws and Institute standards that safeguard privacy, prevent hazing, and allow everyone to exercise "
            "their rights responsibly."
        ),
        keywords=KEYWORD_MAP["sbr-021"],
        reference="student_handbook.txt:288-303",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-022",
        question="When can off-campus misconduct lead to Institute discipline?",
        answer=(
            "Although off-campus misconduct is usually outside Institute discipline, it will be reviewed when it threatens "
            "people or property within the Rensselaer community, and the Institute regulates private conduct that hazards "
            "others' rights, breaks laws, or disrupts academic and administrative processes."
        ),
        keywords=KEYWORD_MAP["sbr-022"],
        reference="student_handbook.txt:304-309",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-023",
        question="What responsibility do recognized student organizations accept?",
        answer=(
            "Institute-recognized organizations accept corporate responsibility to protect community members and guests "
            "during group and individual member activities, designate officers, and understand that officers can face "
            "individual discipline if they fail to uphold policy."
        ),
        keywords=KEYWORD_MAP["sbr-023"],
        reference="student_handbook.txt:310-316",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-024",
        question="Who administers student disciplinary and judicial processes?",
        answer=(
            "Authority for administering student disciplinary and judicial processes resides exclusively with the Dean of "
            "Students Office by delegation from the President."
        ),
        keywords=KEYWORD_MAP["sbr-024"],
        reference="student_handbook.txt:319-320",
    ),
    GroundTruthCase(
        id="sbr-025",
        question="What is the first listed Ground for Disciplinary Action?",
        answer=(
            "Any conduct that could be construed as violating federal, state, or local law constitutes grounds for "
            "disciplinary action."
        ),
        keywords=KEYWORD_MAP["sbr-025"],
        reference="student_handbook.txt:332-334",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-026",
        question="How does the handbook address disruptive conduct?",
        answer=(
            "Conduct that disrupts or interferes with the personal or group rights of community members or with Institute "
            "activities, including access to facilities and normal duties, is a violation."
        ),
        keywords=KEYWORD_MAP["sbr-026"],
        reference="student_handbook.txt:342-344",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-027",
        question="What behaviors violate the rule against unlawful intrusion or seizure?",
        answer=(
            "The handbook prohibits theft or possession of stolen property, using unauthorized Institute keys or access "
            "devices, unauthorized entry, and refusing to leave or surrender property when ordered."
        ),
        keywords=KEYWORD_MAP["sbr-027"],
        reference="student_handbook.txt:345-348",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-028",
        question="How is property damage treated?",
        answer=(
            "Damage to property, including vandalism, is a ground for disciplinary action."
        ),
        keywords=KEYWORD_MAP["sbr-028"],
        reference="student_handbook.txt:349",
    ),
    GroundTruthCase(
        id="sbr-029",
        question="What is said about academic dishonesty in the Grounds for Disciplinary Action?",
        answer=(
            "Academic dishonesty, as defined elsewhere in the handbook, is explicitly listed as a violation."
        ),
        keywords=KEYWORD_MAP["sbr-029"],
        reference="student_handbook.txt:350",
    ),
    GroundTruthCase(
        id="sbr-030",
        question="How does the handbook define fraud as a violation?",
        answer=(
            "Fraud—including forgery, misuse, or alteration of Institute records, documents, or identification—is prohibited."
        ),
        keywords=KEYWORD_MAP["sbr-030"],
        reference="student_handbook.txt:351-352",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-031",
        question="What is the policy on controlled substances?",
        answer=(
            "Using, possessing, or distributing illegal drugs, precursors, or drug paraphernalia except as expressly permitted "
            "by law and Institute regulations is a disciplinary violation."
        ),
        keywords=KEYWORD_MAP["sbr-031"],
        reference="student_handbook.txt:353-355",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-032",
        question="How is disorderly or indecent conduct addressed?",
        answer=(
            "Disorderly, lewd, harassing, or indecent conduct violates the Grounds for Disciplinary Action."
        ),
        keywords=KEYWORD_MAP["sbr-032"],
        reference="student_handbook.txt:357",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-033",
        question="What about physical assaults or threats?",
        answer=(
            "Physical assault or conduct that threatens, encourages, or causes physical harm to people or property is "
            "explicitly prohibited."
        ),
        keywords=KEYWORD_MAP["sbr-033"],
        reference="student_handbook.txt:358",
    ),
    GroundTruthCase(
        id="sbr-034",
        question="How is hazing treated in the handbook?",
        answer=(
            "Hazing, whether defined by the handbook or New York State, is listed as a disciplinary violation."
        ),
        keywords=KEYWORD_MAP["sbr-034"],
        reference="student_handbook.txt:359",
    ),
    GroundTruthCase(
        id="sbr-035",
        question="What expectations exist about cooperating with judicial proceedings?",
        answer=(
            "Willfully refusing to testify after being directed to appear—unless doing so would self-incriminate—or "
            "knowingly providing false testimony or evidence in Institute disciplinary proceedings is a violation."
        ),
        keywords=KEYWORD_MAP["sbr-035"],
        reference="student_handbook.txt:360-369",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-036",
        question="How does the handbook define conduct that endangers safety?",
        answer=(
            "Endangering the safety of the community includes tampering with fire alarms or firefighting equipment, setting "
            "fires on Institute property, using cooking equipment in unauthorized residence hall areas, or recklessly "
            "operating a motor vehicle."
        ),
        keywords=KEYWORD_MAP["sbr-036"],
        reference="student_handbook.txt:370-373",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-037",
        question="What restrictions exist on weapons or dangerous materials?",
        answer=(
            "The use, possession, or storage of dangerous weapons, chemicals, or explosive devices—including firearms, "
            "prohibited knives, ammunition, slingshots, bows, fireworks, or bombs—is forbidden."
        ),
        keywords=KEYWORD_MAP["sbr-037"],
        reference="student_handbook.txt:374-377",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-038",
        question="What is required when interacting with Institute officials?",
        answer=(
            "Failing to comply with Institute officials in the performance of their duties—including failing to present valid "
            "identification or knowingly providing false information—is a violation."
        ),
        keywords=KEYWORD_MAP["sbr-038"],
        reference="student_handbook.txt:378-379",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-039",
        question="How is discriminatory harassment defined as a violation?",
        answer=(
            "Engaging in conduct that harms, threatens, demeans, intimidates, or creates a hostile environment for "
            "protected individuals or groups under the Non-Discrimination policy constitutes discriminatory harassment."
        ),
        keywords=KEYWORD_MAP["sbr-039"],
        reference="student_handbook.txt:385-387",
    ),
    GroundTruthCase(
        id="sbr-040",
        question="When is off-campus conduct considered Institute-related because of the victim?",
        answer=(
            "If the victim is a student or Rensselaer-affiliated individual or group, including the Institute itself, the matter is "
            "Institute-related—even if the alleged violator did not know the victim's status."
        ),
        keywords=KEYWORD_MAP["sbr-040"],
        reference="student_handbook.txt:403-408",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-041",
        question="How does using one's student status affect jurisdiction?",
        answer=(
            "If an alleged violator uses their status as a student or Rensselaer group to facilitate an offense, the off-campus "
            "conduct falls under Institute jurisdiction."
        ),
        keywords=KEYWORD_MAP["sbr-041"],
        reference="student_handbook.txt:409-410",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-042",
        question="What role do Institute-sponsored events play in jurisdiction?",
        answer=(
            "Violations that occur during Rensselaer-sponsored or sanctioned events, or events organized or sponsored by "
            "student organizations, are considered Institute-related."
        ),
        keywords=KEYWORD_MAP["sbr-042"],
        reference="student_handbook.txt:411-412",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-043",
        question="How does serious off-campus misconduct fall under Institute jurisdiction?",
        answer=(
            "Serious infractions likely to cause severe damage to Rensselaer's reputation, threaten community health or "
            "safety, or otherwise endanger members or property—including acts of violence or distributing illegal drugs or "
            "weapons—are treated as Institute-related."
        ),
        keywords=KEYWORD_MAP["sbr-043"],
        reference="student_handbook.txt:413-417",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-044",
        question="What standard of proof does the Rensselaer judicial system use?",
        answer=(
            "Campus disciplinary proceedings apply a preponderance of the evidence standard."
        ),
        keywords=KEYWORD_MAP["sbr-044"],
        reference="student_handbook.txt:420-424",
    ),
    GroundTruthCase(
        id="sbr-045",
        question="What obligations do students have during a judicial inquiry?",
        answer=(
            "When summoned, students must appear, are informed of the concerns, may consult a Student Judicial Adviser "
            "before speaking, must answer questions fully and truthfully (with limited refusal rights to avoid self-incrimination), "
            "and may bring an adviser only with the hearing officer's permission."
        ),
        keywords=KEYWORD_MAP["sbr-045"],
        reference="student_handbook.txt:432-447",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-046",
        question="How can a student request a Judicial Board hearing?",
        answer=(
            "A student who rejects the inquiry decision must send a written request to the Judicial Board Chair via the Senior "
            "Judicial Administrator within three Institute business days; timely requests defer the inquiry sanctions until the "
            "Board issues a decision that overrides the inquiry outcome."
        ),
        keywords=KEYWORD_MAP["sbr-046"],
        reference="student_handbook.txt:463-471",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-047",
        question="What are the only grounds for appealing a judicial decision?",
        answer=(
            "Appeals must cite procedural error, new evidence that could not have been discovered earlier and would likely "
            "change the result, or sanctions that are not appropriate for the violations."
        ),
        keywords=KEYWORD_MAP["sbr-047"],
        reference="student_handbook.txt:472-488",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-048",
        question="How does the academic integrity policy define academic fraud?",
        answer=(
            "Academic fraud is altering documentation related to the grading process, such as changing exam solutions to "
            "argue for a higher grade or tampering with an instructor's grade book."
        ),
        keywords=KEYWORD_MAP["sbr-048"],
        reference="student_handbook.txt:719-721",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-049",
        question="What is the handbook's definition of plagiarism?",
        answer=(
            "Plagiarism means representing another person's work or words as one's own by omitting acknowledgment, for "
            "example copying sentences verbatim, presenting another author's argument, or using digital graphics without "
            "citing the source."
        ),
        keywords=KEYWORD_MAP["sbr-049"],
        reference="student_handbook.txt:738-742",
        pass_threshold=0.67,
    ),
    GroundTruthCase(
        id="sbr-050",
        question="What protection does the Good Samaritan Policy give to a person who receives medical help?",
        answer=(
            "An individual who receives emergency assistance or medical treatment because of alcohol or drug consumption "
            "and completes the assigned assessment, education, and/or treatment through the Health Center will not face "
            "judicial action for violating the Institute's Alcohol and Other Drugs Policy."
        ),
        keywords=KEYWORD_MAP["sbr-050"],
        reference="student_handbook.txt:1434-1439",
        pass_threshold=0.67,
    ),
]
