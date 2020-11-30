<!-- markdownlint-disable MD024 -->
# Contribution Guidelines

Thanks for your interest in contributing to our project. This page will give you a quick overview of how things are organized and, most importantly, how to get involved. Everyone is welcome to contribute, and we value everybody's contribution.

## Table of contents

1. [Add a project](#add-a-project)
2. [Update a project](#update-a-project)
3. [Improve metadata collection](#improve-metadata-collection)
4. [Improve markdown generation](#improve-markdown-generation)
5. [Code of conduct](#code-of-conduct)

## Add a project

If you like to suggest or add a project, choose one of the following ways:

- Suggest a project by opening an issue: Please use the suggest project template from the [issue page](https://github.com/ml-tooling/best-of-ml-python/issues/new/choose) and fill in the requested information.
- Add a project from the Github UI: Edit the [projects.yaml](https://github.com/ml-tooling/best-of-ml-python/edit/main/projects.yaml) file directly on Github and create a pull request with your additions.
- Add a project by forking the repo: Fork this repository, clone it to your computer, modify the `projects.yaml` file, and submit a pull request.

Before opening an issue or pull request, please ensure that you adhere to the following guidelines:

- Please make sure that the project was not already added or suggested to this best-of list. You can ensure this by searching the projects.yaml, the Readme, and the issue list.
- Add the project to the `projects.yaml` and never to the `README.md` file directly. Use the yaml format and the properties documented in the [project properties](#project-properties) section below to add a new project, for example:
    ```yaml
    - name: Tensorflow
      github_id: tensorflow/tensorflow
      category: ml-frameworks
      pypi_id: tensorflow
      conda_id: tensorflow
      docs_url: https://www.tensorflow.org/overview
      labels: ["tensorflow"]
    ```
- Please create an individual issue or pull request for each project.
- Please use the following title format for the issue or pull request: `Add project: project-name`.
- If a project doesn't fit any of the pre-existing categories, it should go under `Others` category by not assigning any category. You can also suggest a new category via the add or update category template on the [issue page](https://github.com/ml-tooling/best-of-ml-python/issues/new/choose).

## Update a project

If you like to suggest or contribute a project update, choose one of the following ways:

- Suggest a project update by opening an issue: Please use the update project template from the [issue page](https://github.com/ml-tooling/best-of-ml-python/issues/new/choose) and fill in the requested information.
- Update the project from the Github UI: Edit the [projects.yaml](https://github.com/ml-tooling/best-of-ml-python/edit/main/projects.yaml) file directly on Github and create a pull request with your changes.
- Update project by forking the repo: Fork this repository, clone it to your computer, modify the `projects.yaml` file, and submit a pull request.

Before opening an issue or pull request, please ensure that you adhere to the following guidelines:

- Only update the project in the `projects.yaml` and never to the `README.md` file directly. Use the yaml format and the properties documented in the [project properties](#project-properties) section below to update a new project.
- Please create an individual issue or pull request for each project.
- Please use the following title format for the issue or pull request: `Update project: project-name`.

## Project properties

<table>
    <tr>
        <th>Property</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>name</td>
        <td>Name of the project.</td>
    </tr>
    <tr>
        <td>github_id</td>
        <td>Github ID of the project based on user or organization  and the repository name (e.g. <code>ml-tooling/best-of-generator</code>).</td>
    </tr>
    <tr>
        <td>category</td>
        <td>Category that this project is most related to. You can find all available category IDs in the <code>projects.yaml</code> file. The project will be sorted into the <code>Others</code> category if no category is provided.</td>
    </tr>
    <tr>
        <td colspan="2"><b>Optional Properties:</b></td>
    </tr>
    <tr>
        <td>license</td>
        <td>License of the project. If set, license information from Github or package managers will be overwritten.</td>
    </tr>
    <tr>
        <td>labels</td>
        <td>List of labels that this project is related to. You can find all available label IDs in the projects.yaml file.</td>
    </tr>
    <tr>
        <td>description</td>
        <td>Short description of the project. If set, the description from Github or package managers will be overwritten.</td>
    </tr>
    <tr>
        <td>homepage</td>
        <td>Hompage URL of the project. Only use this property if the project homepage is different from the Github URL.</td>
    </tr>
    <tr>
        <td>docs_url</td>
        <td>Documentation URL of the project. Only use this property if the project documentation site is different from the Github URL.</td>
    </tr>
    <tr>
        <td colspan="2"><b>Supported Package Managers:</b></td>
    </tr>
    <tr>
        <td>pypi_id</td>
        <td>Project ID on the python package index (pypi.org).</td>
    </tr>
    <tr>
        <td>conda_id</td>
        <td>Project ID on the conda package manager (anaconda.org). If the main package is provided on a different channel, prefix the ID with the given channel: e.g. <code>conda-forge/tensorflow</code></td>
    </tr>
    <tr>
        <td>npm_id</td>
        <td>Project ID on the Node package manager (npmjs.com).</td>
    </tr>
    <tr>
        <td>dockerhub_id</td>
        <td>Project ID on the Dockerhub container registry (hub.docker.com). </td>
    </tr>
</table>

## Improve metadata collection

If you like to contribute to or share suggestions regarding the project metadata collection, please refer to the [best-of-generator](https://github.com/ml-tooling/best-of-generator) repository.

## Improve markdown generation

If you like to contribute to or share suggestions regarding the markdown generation, please refer to the [best-of-generator](https://github.com/ml-tooling/best-of-generator) repository.

## Code of Conduct

All members of the project community must abide by the [Contributor Covenant, version 2.0](./.github/CODE_OF_CONDUCT.md). Only by respecting each other we can develop a productive, collaborative community. Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting a project maintainer.
