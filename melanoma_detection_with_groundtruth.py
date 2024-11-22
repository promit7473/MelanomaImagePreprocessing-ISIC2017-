def calculate_metrics(pred_mask, ground_truth_mask):
    """Calculate Dice coefficient and IoU for segmentation evaluation"""
    smooth = 1e-6
    
    pred_mask = pred_mask.flatten()
    ground_truth_mask = ground_truth_mask.flatten()
    
    intersection = np.sum(pred_mask * ground_truth_mask)
    
    # Dice coefficient
    dice = (2. * intersection + smooth) / (np.sum(pred_mask) + np.sum(ground_truth_mask) + smooth)
    
    # IoU (Intersection over Union)
    iou = (intersection + smooth) / (np.sum(pred_mask) + np.sum(ground_truth_mask) - intersection + smooth)
    
    return dice, iou

def evaluate_predictions(model, test_loader, device, ground_truth_dir):
    model.eval()
    dice_scores = []
    iou_scores = []
    
    # Create directories for visualization
    os.makedirs('evaluation_results', exist_ok=True)
    os.makedirs('evaluation_results/predictions', exist_ok=True)
    os.makedirs('evaluation_results/comparison', exist_ok=True)
    
    with torch.no_grad():
        for i, (images, ground_truth_masks) in enumerate(test_loader):
            # Get predictions
            images = images.to(device)
            outputs = model(images)
            pred_masks = (outputs > 0.5).float().cpu().numpy()
            
            # Convert ground truth to numpy
            ground_truth_masks = ground_truth_masks.numpy()
            
            # Calculate metrics for each image in batch
            for j, (pred, truth) in enumerate(zip(pred_masks, ground_truth_masks)):
                dice, iou = calculate_metrics(pred[0], truth[0])
                dice_scores.append(dice)
                iou_scores.append(iou)
                
                # Save visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image (denormalize first)
                img = images[j].cpu().numpy().transpose(1, 2, 0)
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                img = np.clip(img, 0, 1)
                
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Predicted mask
                axes[1].imshow(pred[0], cmap='gray')
                axes[1].set_title(f'Predicted Mask\nDice: {dice:.3f}, IoU: {iou:.3f}')
                axes[1].axis('off')
                
                # Ground truth mask
                axes[2].imshow(truth[0], cmap='gray')
                axes[2].set_title('Ground Truth Mask')
                axes[2].axis('off')
                
                plt.savefig(f'evaluation_results/comparison/comparison_{i}_{j}.png')
                plt.close()
    
    # Calculate and save overall metrics
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    metrics_summary = {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'std_dice': np.std(dice_scores),
        'std_iou': np.std(iou_scores)
    }
    
    # Plot histogram of scores
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(dice_scores, bins=20)
    plt.title(f'Dice Scores (Mean: {mean_dice:.3f})')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(iou_scores, bins=20)
    plt.title(f'IoU Scores (Mean: {mean_iou:.3f})')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/score_distribution.png')
    plt.close()
    
    return metrics_summary

# Add this to your main function after training
def evaluate_model():
    print("\nEvaluating model against ground truth...")
    
    # Load your best model
    model = UNet().to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    
    # Create test dataset and loader
    test_dataset = MelanomaDataset(test_img_paths, test_mask_paths, transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Run evaluation
    metrics = evaluate_predictions(model, test_loader, device, ground_truth_dir)
    
    print("\nEvaluation Results:")
    print(f"Mean Dice Coefficient: {metrics['mean_dice']:.3f} ± {metrics['std_dice']:.3f}")
    print(f"Mean IoU: {metrics['mean_iou']:.3f} ± {metrics['std_iou']:.3f}")
    print("\nDetailed results and visualizations saved in 'evaluation_results' directory")
