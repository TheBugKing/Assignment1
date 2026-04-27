# RAG NXP test run
Endpoint: `http://localhost:8000/query`
Filter: tiers=['T1', 'T2', 'T3', 'T4', 'T5'] limit=none (14 of 14 questions)

## T1.1 [PASS] (118.1s) - T1 basic
**Q:** What are the six goals stated in the Preamble?
**A:** Establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity.
**Verdict reason:** matched: ['establish justice', 'justice', 'domestic tranquility', 'tranquility', 'common defence', 'defence', 'general welfare', 'welfare', 'blessings of liberty', 'liberty']

**Retrieved chunks (page hints):**
- [1] Preamble, page 1 — dist=1.3762904405593872
- [2] Article I - Section 8, page 6 — dist=1.2111934423446655
- [3] Article I - Section 8, page 5 — dist=1.3503228425979614
- [4] Article VI, page 14 — dist=1.352055549621582
- [5] Article II - Section 1, page 9 — dist=1.3661422729492188
- [6] Article V, page 14 — dist=1.3769534826278687
- [7] Article IV - Section 4, page 14 — dist=1.3882057666778564
- [8] Article I - Section 2, page 1 — dist=1.4163306951522827

## T1.2 [PASS] (119.7s) - T1 basic
**Q:** How many years is a Senator's term?
**A:** Six Years
**Verdict reason:** matched: ['six']

**Retrieved chunks (page hints):**
- [1] Article I - Section 3, page 2 — dist=0.7174116969108582
- [2] Article I - Section 3, page 2 — dist=0.7529313564300537
- [3] Article I - Section 2, page 1 — dist=1.0494744777679443
- [4] Article I - Section 2, page 1 — dist=1.0608468055725098
- [5] Article I - Section 6, page 4 — dist=1.0977740287780762
- [6] Article I - Section 2, page 1 — dist=1.1070294380187988
- [7] Article I - Section 3, page 2 — dist=1.1194179058074951
- [8] Article II - Section 1, page 9 — dist=1.1613118648529053

## T1.3 [PASS] (133.1s) - T1 basic
**Q:** What is the minimum age to become President?
**A:** Person be eligible to that Office who shall not have attained to the Age of thirty five Years, and been fourteen Years a Resident within the United States.
**Verdict reason:** matched: ['thirty five']

**Retrieved chunks (page hints):**
- [1] Article I - Section 3, page 2 — dist=0.9469693899154663
- [2] Article II - Section 1, page 10 — dist=1.0092155933380127
- [3] Article I - Section 2, page 1 — dist=1.0573008060455322
- [4] Article II - Section 1, page 9 — dist=1.0896563529968262
- [5] Article II - Section 1, page 9 — dist=1.238149881362915
- [6] Article I - Section 2, page 1 — dist=1.2905148267745972
- [7] Article II - Section 2, page 10 — dist=1.30539870262146
- [8] Article II - Section 1, page 9 — dist=1.3461320400238037

## T2.4 [PASS] (127.0s) - T2 numerical
**Q:** How many states must ratify a constitutional amendment for it to pass?
**A:** three fourths of the several States, or by Conventions in three fourths thereof, as the one or the other Mode of Ratification may be proposed by the Congress;
**Verdict reason:** matched: ['three fourths']

**Retrieved chunks (page hints):**
- [1] Article V, page 14 — dist=0.6981327533721924
- [2] Article I - Section 2, page 1 — dist=0.8574821949005127
- [3] Article IV - Section 3, page 13 — dist=0.9145797491073608
- [4] Article VI, page 14 — dist=0.9877535104751587
- [5] Article IV - Section 4, page 14 — dist=1.0662355422973633
- [6] Article I - Section 8, page 7 — dist=1.1031789779663086
- [7] Article VII, page 15 — dist=1.135374903678894
- [8] Article II - Section 1, page 9 — dist=1.145067811012268

## T2.5 [PASS] (132.9s) - T2 numerical
**Q:** How many senators does each state get?
**A:** Each State has two Senators.
**Verdict reason:** matched: ['two']

**Retrieved chunks (page hints):**
- [1] Article II - Section 1, page 9 — dist=0.9346451163291931
- [2] Article I - Section 2, page 1 — dist=0.9450455904006958
- [3] Article I - Section 3, page 2 — dist=0.9688328504562378
- [4] Article II - Section 1, page 9 — dist=0.990908682346344
- [5] Article I - Section 2, page 1 — dist=1.0837385654449463
- [6] Article I - Section 3, page 2 — dist=1.122253179550171
- [7] Article I - Section 2, page 1 — dist=1.1639304161071777
- [8] Article I - Section 4, page 3 — dist=1.1912147998809814

## T2.6 [PASS] (162.7s) - T2 numerical
**Q:** What fraction of Congress is needed to override a presidential veto?
**A:** Two thirds of both Houses.
**Verdict reason:** matched: ['two thirds']

**Retrieved chunks (page hints):**
- [1] Article V, page 14 — dist=1.04617440700531
- [2] Article I - Section 7, page 4 — dist=1.0686352252960205
- [3] Article I - Section 7, page 4 — dist=1.0851935148239136
- [4] Article II - Section 1, page 9 — dist=1.1011323928833008
- [5] Article I - Section 7, page 5 — dist=1.1386457681655884
- [6] Article I - Section 8, page 5 — dist=1.1545222997665405
- [7] Article I - Section 2, page 1 — dist=1.1905810832977295
- [8] Article II - Section 4, page 11 — dist=1.1988005638122559

## T3.7 [PASS] (108.7s) - T3 clause
**Q:** Which house has the sole power of impeachment?
**A:** The House of Representatives has the sole Power of Impeachment.
**Verdict reason:** matched: ['house of representatives', 'representatives']

**Retrieved chunks (page hints):**
- [1] Article I - Section 3, page 3 — dist=0.7806759476661682
- [2] Article II - Section 4, page 11 — dist=0.9386943578720093
- [3] Article I - Section 2, page 2 — dist=0.9526465535163879
- [4] Article II - Section 2, page 10 — dist=1.1458677053451538
- [5] Article I - Section 5, page 4 — dist=1.1763654947280884
- [6] Article I - Section 7, page 4 — dist=1.1856528520584106
- [7] Article I - Section 5, page 3 — dist=1.1933577060699463
- [8] Article I - Section 6, page 4 — dist=1.2255656719207764

## T3.8 [PASS] (74.5s) - T3 clause
**Q:** Who presides over the Senate when the President is on trial during impeachment?
**A:** The Chief Justice shall preside.
**Verdict reason:** matched: ['chief justice']

**Retrieved chunks (page hints):**
- [1] Article I - Section 3, page 3 — dist=0.5384433269500732
- [2] Article II - Section 4, page 11 — dist=0.8927856683731079
- [3] Article II - Section 2, page 10 — dist=0.983344316482544
- [4] Article I - Section 3, page 2 — dist=1.0109496116638184
- [5] Article III - Section 2, page 12 — dist=1.0371968746185303
- [6] Article II - Section 2, page 11 — dist=1.0374870300292969
- [7] Article I - Section 6, page 4 — dist=1.0489104986190796
