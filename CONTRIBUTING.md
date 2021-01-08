<!-- markdownlint-disable MD024 -->
# Contribution Guidelines

Thanks for your interest in contributing to our project. This page will give you a quick overview of how things are organized and, most importantly, how to get involved. Everyone is welcome to contribute, and we value everybody's contribution.

## Table of contents

1. [Add a project](#add-a-project)
2. [Update a project](#update-a-project)
3. [Improve metadata collection](#improve-metadata-collection)
4. [Improve markdown generation](#improve-markdown-generation)
5. [Create your own best-of list](#improve-markdown-generation)
6. [Code of conduct](#code-of-conduct)

## Add a project

If you like to suggest or add a project, choose one of the following ways:

- Suggest a project by opening an issue: Please use the suggest project template from the [issue page](https://github.com/ml-tooling/best-of-ml-python/issues/new/choose) and fill in the requested information.
- Add a project by modifying the [projects.yaml](https://github.com/ml-tooling/best-of-ml-python/blob/main/projects.yaml) and submitting a pull request with your addition. This can also be done directly via the [Github UI](https://github.com/ml-tooling/best-of-ml-python/edit/main/projects.yaml).

Before opening an issue or pull request, please ensure that you adhere to the following guidelines:

- Please make sure that the project was not already added or suggested to this best-of list. You can ensure this by searching the projects.yaml, the Readme, and the issue list.
- Add the project to the `projects.yaml` and never to the `README.md` file directly. Use the yaml format and the properties documented in the [project properties](#project-properties) section below to add a new project, for example:
    ```yaml
    - name: Tensorflow
      github_id: tensorflow/tensorflow
      pypi_id: tensorflow
      conda_id: tensorflow
      labels: ["tensorflow"]
      category: ml-frameworks
    ```
- Please create an individual issue or pull request for each project.
- Please use the following title format for the issue or pull request: `Add project: project-name`.
- If a project doesn't fit into any of the pre-existing categories, it should go under the `Others` category by not assigning any category. You can also suggest a new category via the add or update category template on the [issue page](https://github.com/ml-tooling/best-of-ml-python/issues/new/choose).

## Update a project

If you like to suggest or contribute a project update, choose one of the following ways:

- Suggest a project update by opening an issue: Please use the update project template from the [issue page](https://github.com/ml-tooling/best-of-ml-python/issues/new/choose) and fill in the requested information.
- Update a project by modifying the [projects.yaml](https://github.com/ml-tooling/best-of-ml-python/blob/main/projects.yaml) and submitting a pull request with your changes. This can also be done directly via the [Github UI](https://github.com/ml-tooling/best-of-ml-python/edit/main/projects.yaml).

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
        <td><code>name</code></td>
        <td>Name of the project. This name is required to be unique on the best-of list.</td>
    </tr>
    <tr>
        <td><code>github_id</code></td>
        <td>Github ID of the project based on user or organization and the repository name, e.g. <code>best-of-lists/best-of-generator</code>.</td>
    </tr>
    <tr>
        <td colspan="2"><b>Optional Properties:</b></td>
    </tr>
    <tr>
        <td><code>category</code></td>
        <td>Category that this project is most related to. You can find all available category IDs in the <code>projects.yaml</code> file. The project will be sorted into the <code>Others</code> category if no category is provided.</td>
    </tr>
    <tr>
        <td><code>labels</code></td>
        <td>List of labels that this project is related to. You can find all available label IDs in the <code>projects.yaml</code> file.</td>
    </tr>
    <tr>
        <td colspan="2"><b>Supported Package Managers:</b></td>
    </tr>
    <tr>
        <td><code>pypi_id</code></td>
        <td>Project ID on the python package index (<a href="https://pypi.org">PyPi</a>).</td>
    </tr>
    <tr>
        <td><code>conda_id</code></td>
        <td>Project ID on the <a href="https://anaconda.org">conda package manager</a>. If the main package is provided on a different channel, prefix the ID with the given channel: e.g. <code>conda-forge/tensorflow</code></td>
    </tr>
    <tr>
        <td><code>npm_id</code></td>
        <td>Project ID on the Node package manager (<a href="https://www.npmjs.com">npm</a>).</td>
    </tr>
    <tr>
        <td><code>dockerhub_id</code></td>
        <td>Project ID on the <a href="https://hub.docker.com">Docker Hub container registry</a>. </td>
    </tr>
    <tr>
        <td><code>maven_id</code></td>
        <td>Artifact ID on <a href="https://mvnrepository.com">Maven central</a>, e.g. <code>org.apache.flink:flink-core</code>. </td>
    </tr>
</table>

Please refer to the [best-of-generator documentation](https://github.com/best-of-lists/best-of-generator#project-properties) for a complete and up-to-date list of supported project properties.

## Improve metadata collection

If you like to contribute to or share suggestions regarding the project metadata collection, please refer to the [best-of-generator](https://github.com/best-of-lists/best-of-generator) repository.

## Improve markdown generation

If you like to contribute to or share suggestions regarding the markdown generation, please refer to the [best-of-generator](https://github.com/best-of-lists/best-of-generator) repository.

## Create your own best-of list

If you want to create your own best-of list, we strongly recommend to follow [this guide](https://github.com/best-of-lists/best-of/blob/main/create-best-of-list.md). With this guide, it will only take about 3 minutes to get you started. It is already set-up to automatically run the best-of generator via our Github Action and includes other useful template files.

## Code of Conduct

All members of the project community must abide by the [Contributor Covenant, version 2.0](./.github/CODE_OF_CONDUCT.md). Only by respecting each other we can develop a productive, collaborative community. Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting a project maintainer.
