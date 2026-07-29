---
doc_id: ENG-ONB-02
title: Engineering Onboarding Guide
version: "2"
status: current
department: Engineering
access: all-staff
effective_date: 2026-02-01
owner: Chong Wei Ming
---

# Engineering Onboarding Guide

For new engineers joining the Shah Alam design office. Work through this
in your first four weeks with your assigned buddy.

## Week 1 — Access and Orientation

### 1.1 Accounts

Your line manager raises the access request; IT provisions within 2
working days. You should have:

- Email and Microsoft 365
- ERP access, read-only initially, transactional access after training
- CAD vault access, read and check-out on your assigned project areas
- Test data repository, read access
- VPN profile for remote access

Multi-factor authentication is mandatory on all of the above. Set it up on
day one; you will be locked out otherwise.

### 1.2 Site Induction

Before entering the factory or test bay you must complete the HSE
induction with the Safety Officer. Until then you may not go past the
turnstile into production areas, even accompanied.

Collect your safety footwear and safety glasses from Stores. Hearing
protection is issued at the test bay entrance.

### 1.3 Reading

In your first week, read:

1. The Employee Handbook
2. SOP-SAF-07, the lockout-tagout procedure
3. The specification for the product family you are assigned to
4. The compatibility matrix, ENG-COMPAT-11
5. The end-of-life register, ENG-EOL-05

## Week 2 — Engineering Standards

### 2.1 Drawing Standards

- Projection: first angle, per ISO 128
- Units: millimetres, three decimal places on machined features
- Tolerancing: geometric dimensioning and tolerancing per ISO 1101
- Surface finish: Ra in micrometres
- Title block: use the current template only, `NWH-TITLE-2024.dwt`
- Drawing numbers are issued from the register; never invent one

### 2.2 Revision Control

Drawings are revisioned A, B, C. A revision letter is only issued after an
approved engineering change note. Never edit a released drawing without an
ECN; work in a check-out copy.

### 2.3 CAD Vault Discipline

Check out, work, check in the same day where possible. Never keep files
checked out over a weekend without telling your team, as it blocks others.
Never copy vault files to a local drive or a personal cloud account.

## Week 3 — The Engineering Change Process

### 3.1 When an ECN Is Required

Any change to a released drawing, bill of material, material
specification, supplier of a critical part, or test procedure.

### 3.2 Process

1. Raise form **ENG-ECN-03** with the reason, affected part numbers and a
   description of the change.
2. Assess impact: cost, tooling, existing stock, units in the field,
   certification, and whether the change is interchangeable.
3. Obtain reviews. A change affecting a marine product always requires
   Quality Manager review because of type approval implications.
4. Obtain approval. Changes below RM 10,000 impact are approved by the
   Head of Engineering; above that, by the Management Committee.
5. Set the effective point: from a serial number, from a date, or from
   stock exhaustion. State which.
6. Update the drawing, the bill of material and the service documentation
   together. An ECN is not closed until all three are done.

### 3.3 Target Timescale

ECNs should be closed within 30 days. Anything open beyond 60 days is
reported to the Head of Engineering and must be closed or formally
re-baselined with a stated reason.

## Week 4 — Design Reviews and Test

### 4.1 Design Review Gates

| Gate | Purpose | Chair |
|---|---|---|
| DR0 | Requirements agreed, business case | Head of Engineering |
| DR1 | Concept selected, risks identified | Head of Engineering |
| DR2 | Detailed design complete, DFMEA closed | Head of Engineering |
| DR3 | Prototype test results reviewed | Quality Manager |
| DR4 | Production readiness, release to manufacture | Management Committee |

No gate may be skipped. A gate may be passed with actions, but actions
must be closed before the next gate.

### 4.2 Test Bay

Rigs HTB-1 to HTB-6 are booked through the test schedule; HTB-7 is
expected on site in August 2026. Book at least 5 working days ahead.

You may not operate a rig unsupervised until signed off by the Test
Engineer. Any work inside a rig guard requires lockout-tagout, and you
must be a certified LOTO operator to apply a lock.

### 4.3 Test Data

All test data is uploaded to the test repository on the day it is taken,
with the rig identifier, the unit serial number, the fluid batch and the
ambient conditions recorded. Data without this context is not admissible
in a design review.

## Ongoing

### 5.1 Your Buddy and Your Manager

Your buddy answers day-to-day questions. Your manager holds a weekly
one-to-one for your first three months. Probation is 3 months, and
confirmation depends on a written assessment.

### 5.2 Training Budget

You have an annual training budget of RM 2,000, approved by the Head of
Engineering. Professional body membership for one body is reimbursed up to
RM 600 a year. Use form HR-TR-08.

### 5.3 Weekly Sync

Engineering meets every Monday at 14:00. Bring your open actions. During
active design review cycles, Friday is an on-site day for the whole
department.

### 5.4 Common Mistakes New Engineers Make

- Quoting a marine pump with an NG10 directional valve and forgetting the
  adapter kit
- Assuming the NH-8840 and NH-8840-X share a pressure rating; they do not
- Assuming a discontinued model has no spares without checking the
  end-of-life register
- Editing a released drawing without an ECN
- Booking test bay time without checking whether a LOTO certified person
  is available
