"""Sends tasks to AMT"""
import boto3
import csv
import urllib
import random

# This was written with a different configuration than I typically use.
# And I have no qualms about disabling no-member
# pylint: disable=bad-indentation, no-member

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
url = "SET THIS"

# Load keys
with open("iam-private.txt", "r") as iam_file:
    iam_private = iam_file.readline()[:-1]

with open("iam-public.txt", "r") as iam_file:
    iam_public = iam_file.readline()[:-1]

image_locs = []
seed_descs = []
names = []
target_urls = []

# Loop through all videos.
with open("bbox-seed-descriptions.tsv", "r") as seed_description:
    reader = csv.DictReader(seed_description, delimiter="\t")
    for row in reader:
        # Create a HIT
        names.append(row['Name'])
        seed_descs.append(row['Description'])
        image_locs.append(row['Image File'])
        target_urls.append(url+
                           "?image=images/"+image_locs[-1]+
                           "&amp;object="+urllib.parse.quote(seed_descs[-1])+
                           "&amp;name="+names[-1])

qualifications = [{'QualificationTypeId': '00000000000000000071',
                   'Comparator': 'EqualTo',
                   'LocaleValues': [{'Country': 'US'}],
                   'RequiredToPreview': True}]

mturk = boto3.client('mturk',
                     aws_access_key_id=iam_public,
                     aws_secret_access_key=iam_private,
                     region_name='us-east-1',
                     endpoint_url=MTURK_SANDBOX)

# Shuffle the HIT order.
random.shuffle(target_urls)
for i in range(len(target_urls)):
    print(target_urls[i])
    question = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
            <ExternalQuestion xmlns=\""+\
    "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas"+\
    "/2006-07-14/ExternalQuestion.xsd\"><ExternalURL>"+\
            target_urls[i]+"</ExternalURL><FrameHeight>0"+\
            "</FrameHeight></ExternalQuestion>"


    # Create the HIT.
    new_hit = mturk.create_hit(
        Title='Bounding Box Annotation',
        Description='Please draw a bounding box around '+\
            'the object indicated by the given text.',
        Keywords='bounding box, bbox, image',
        Reward='0.06',
        MaxAssignments=2,
        LifetimeInSeconds=259200,
        AssignmentDurationInSeconds=600,
        AutoApprovalDelayInSeconds=172800,
        Question=question,
        QualificationRequirements=qualifications,
    )

    print("A new HIT has been created. You can preview it here:")
    print("https://workersandbox.mturk.com/mturk/preview?groupId=" +\
          new_hit['HIT']['HITGroupId'])
    print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")

