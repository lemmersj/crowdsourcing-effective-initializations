"""Retrieves initializations from Mechanical Turk"""
import boto3
import time
import string
import random
from IPython import embed
import csv
import os

# This was written with a different configuration than I typically use.
# And I have no qualms about disabling no-member
# pylint: disable=bad-indentation, no-member

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

with open("../iam-private.txt", "r") as iam_file:
    iam_private = iam_file.readline()[:-1]

with open("../iam-public.txt", "r") as iam_file:
    iam_public = iam_file.readline()[:-1]

mturk = boto3.client('mturk',
                     aws_access_key_id=iam_public,
                     aws_secret_access_key=iam_private,
                     region_name='us-east-1',
                     endpoint_url=MTURK_SANDBOX)

# Get reviewable HITs
mturk_returned = mturk.list_reviewable_hits()
hit_dict = mturk_returned['HITs']
while "NextToken" in mturk_returned.keys():
    mturk_returned = mturk.list_reviewable_hits(
        NextToken=mturk_returned['NextToken'])
    hit_dict = hit_dict+mturk_returned['HITs']

to_write = []
ids_to_approve = []
ids_to_reject = []
turker_id_dict = {}
letters = string.ascii_letters+string.digits
# For every HIT (i.e., every initializing bbox)
for hit in hit_dict:
    time.sleep(0.25)
    mturk_response = mturk.list_assignments_for_hit(HITId=hit['HITId'])
    # Get every assignment that is submitted, but not yet approved.
    for assignment in mturk_response['Assignments']:
        if not assignment['AssignmentStatus'] == 'Submitted' and not \
           assignment['AssignmentStatus'] == 'Approved':
            continue
        # String manipulation.
        # Quick and dirty way to chop out what we need
        split_on_freetext = assignment['Answer'].split("<FreeText>")
        # If the split fails, it's probably because no answer was given.
        # and the task should be rejected.
        # But double check.
        if len(split_on_freetext) == 1:
            print("No answer given. Confirm.")
            print(assignment['Answer'])
            embed()
            if assignment['AssignmentStatus'] == 'Submitted':
                ids_to_reject.append(assignment['AssignmentId'])
            continue
        else:
            # Otherwise, save the answer.
            answer = eval(split_on_freetext[1].split("</FreeText>")[0])
            if not answer == []:
                if assignment['AssignmentStatus'] == 'Submitted':
                    ids_to_approve.append(assignment['AssignmentId'])
            else:
                print("No answer given. Confirm.")
                print(assignment['Answer'])
                embed()
                if assignment['AssignmentStatus'] == 'Submitted':
                    ids_to_reject.append(assignment['AssignmentId'])
                continue

        mturk.associate_qualification_with_worker(
            # real server
            #QualificationTypeId="3C8RUX4LEWZS0GOJNSAS3IJT1XRM9Z",
            # sandbox
            QualificationTypeId="3MCN1A5LKUWI2F4NJ5PMNLAPPQWKP5",
            WorkerId=assignment['WorkerId'],
            IntegerValue=1,
            SendNotification=False)
        assignment.update(answer[0])
        assignment['AcceptTime'] = time.mktime(
            assignment['AcceptTime'].timetuple())
        assignment['SubmitTime'] = time.mktime(
            assignment['SubmitTime'].timetuple())
        assignment['AutoApprovalTime'] = time.mktime(
            assignment['AutoApprovalTime'].timetuple())

        # To comply with privacy rules, a random worker ID is generated
        # that can not be traced back to the corresponding Amazon account.
        if assignment['WorkerId'] not in turker_id_dict.keys():
            turker_id_dict[assignment['WorkerId']] = ''.join(
                random.choices(letters, k=50))

        assignment['RandomID'] = turker_id_dict[assignment['WorkerId']]
        assignment.pop('WorkerId')
        assignment.pop('Answer')
        to_write.append(assignment)
        print(str(len(ids_to_approve)) + " approved")
        print(str(len(ids_to_reject)) + " rejected")

# Create output file
file_exists = os.path.isfile("mturk_output.tsv")
with open("mturk_output.tsv", "a") as mturk_csv:
    writer = csv.DictWriter(mturk_csv,
                            fieldnames=to_write[0].keys(),
                            delimiter="\t")
    if not file_exists:
        writer.writeheader()

    # Write all rows
    for row in to_write:
        writer.writerow(row)

# Approve and reject assignments.
print("Approving Assignments")
for id_to_approve in ids_to_approve:
    time.sleep(0.25)
    mturk.approve_assignment(AssignmentId=id_to_approve)

print("Rejecting Assignments")
for id_to_reject in ids_to_reject:
    time.sleep(0.25)
    mturk.reject_assignment(
        AssignmentId=id_to_reject,
        RequesterFeedback="No bounding box was drawn.")
