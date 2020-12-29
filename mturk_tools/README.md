# Mechanical Turk Tools
Tools used to collect experimental data from mechanical turk. The annotation tool (html, coffee, and javascript files) is from [here](https://github.com/kyamagu/bbox-annotator).

## Creating/Retrieving HITS
1. Place the files annotation_tool.html and bbox_annotator.js on your server.
2. Download the OTB100 dataset (using [these instructions](https://github.com/lemmersj/crowdsourcing-effective-initializations/tree/main/tracker)), and place in the images folder.
3. Set the URL of annotation_tool.html on line 12 in create_tasks.py.
4. Place text files containing public (iam-public.txt) and private (iam-private.txt) in the mturk_tools directory.
5. Run:
	

	    python create_tasks.py

This will create jobs on the sandbox server. To create a live run, remove the endpoint URL, and check the submit target in annotation_tool.html.

To retrieve the annotations, simply run:

    python retrive_tasks.py

Note that this will automatically approve HITs for which a bounding box is drawn, and request confirmation, then reject, HITs for which there is no bounding box.

