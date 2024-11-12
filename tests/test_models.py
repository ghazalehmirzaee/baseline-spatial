# tests/test_models.py

import pytest
import torch
from src.models.integration import GraphAugmentedViT
from src.models.graph.spatial_graph import SpatialDistanceGraph


@pytest.fixture
def model_config():
    return {
        'num_classes': 14,
        'vit_pretrained': False,
        'feature_dim': 768,
        'graph_hidden_dim': 256,
        'graph_num_heads': 8,
        'graph_dropout': 0.1,
        'fusion_type': 'adaptive'
    }


@pytest.fixture
def sample_batch():
    batch_size = 4
    return {
        'images': torch.randn(batch_size, 3, 1000, 1000),
        'labels': torch.randint(0, 2, (batch_size, 14)),
        'bb_coords': torch.randn(batch_size, 14, 4)
    }


def test_model_initialization(model_config):
    model = GraphAugmentedViT(**model_config)
    assert isinstance(model, GraphAugmentedViT)
    assert isinstance(model.spatial_graph, SpatialDistanceGraph)


def test_model_forward(model_config, sample_batch):
    model = GraphAugmentedViT(**model_config)
    outputs = model(
        sample_batch['images'],
        sample_batch['bb_coords'],
        sample_batch['labels']
    )

    assert 'logits' in outputs
    assert outputs['logits'].shape == (4, 14)
    assert 'loss' in outputs
    assert isinstance(outputs['loss'], torch.Tensor)


def test_attention_maps(model_config, sample_batch):
    model = GraphAugmentedViT(**model_config)
    attention_maps = model.get_attention_maps(
        sample_batch['images'],
        sample_batch['bb_coords']
    )

    assert isinstance(attention_maps, dict)
    assert len(attention_maps) > 0


# tests/test_data.py

import pytest
from src.data.datasets import ChestXrayDataset
from pathlib import Path


@pytest.fixture
def dataset_config():
    return {
        'image_dir': '/users/gm00051/ChestX-ray14/categorized_images/train',
        'label_file': '/users/gm00051/ChestX-ray14/labels/train_list.txt',
        'bbox_file': '/users/gm00051/ChestX-ray14/labels/BBox_List_2017.csv'
    }


def test_dataset_initialization(dataset_config):
    dataset = ChestXrayDataset(**dataset_config)
    assert len(dataset) > 0


def test_dataset_getitem(dataset_config):
    dataset = ChestXrayDataset(**dataset_config)
    image, labels, bb_coords = dataset[0]

    assert image.shape == (3, 1000, 1000)
    assert labels.shape == (14,)
    assert bb_coords is None or bb_coords.shape == (14, 4)

