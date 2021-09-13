# Importing Files in an S3 Bucket to LabelBox

## First Method: Generate signed urls

Use the script `gen-signed-url.py` to generate signed urls, but they will expire after 7 days.

## Second Method: Deploying Heroku for Signed URLs

This is a two step process of first deploying a Heroku app and then generating the signed urls.

The following folders `generate-urls` and `heroku-deploy-signed-urls` were added as submodules. Please run: `git submodule update --init --recursive` to ensure the folders are download into the repository.

### Part 1: Deploying Heroku to get static signed URLs
1. If you don't have a Heroku account, you will have to signup first.
2. Enter  `heroku-deploy-signed-urls` directory.
2. Click the `Deploy the Heroku app` button to deploy the app. 
3. It will bring you to Heroku where once you've signed in you will have to enter your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` and click Deploy app.
4. Click `Manage App` and then  `Settings` and then `Reveal Config Vars` to copy the `SECRET` token to enter in Part 2.

### Part 2: Generate a json file with URLs that point at the Heroku proxy (static URLs)
1. Enter the `generate-urls` folder.
2. Ensure you have `node.js` installed by running `node --version`.
3. Run `npm install`.
4. Run `AWS_PROFILE=crayon-site node cli.js 
  --bucket st-crayon
  --prefix <aws-prefix-path-in-bucket>
  --host https://<your-new-heroku-url>.herokuapp.com/
  --secret <heroku-generated-config-secret>
  --output labelbox-import.json` (change the AWS_PROFILE accordingly or remove for the default profile)

All commands for part 2:
```bash
cd generate-urls/

// confirm you have node.js installed
node --version

npm install
AWS_PROFILE=crayon-site node cli.js 
  --bucket st-crayon
  --prefix <aws-prefix-path-in-bucket>
  --host https://<your-new-heroku-url>.herokuapp.com/
  --secret <heroku-generated-config-secret>
  --output labelbox-import.json
```
