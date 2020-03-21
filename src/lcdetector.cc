/*
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#include "ibow-lcd/lcdetector.h"

namespace ibow_lcd{

LCDetector::LCDetector(const LCDetectorParams &params) : last_lc_island_(-1, 0.0, -1, -1)
{
  // Creating the image index
  index_ = std::make_shared<obindex2::ImageIndex>(params.k,
                                                  params.s,
                                                  params.t,
                                                  params.merge_policy,
                                                  params.purge_descriptors,
                                                  params.min_feat_apps);

  index_l_ = std::make_shared<obindex2::ImageIndex>(params.k,
                                                    params.s,
                                                    params.t,
                                                    params.merge_policy,
                                                    params.purge_descriptors,
                                                    params.min_feat_apps);
  // Storing the remaining parameters
  p_ = params.p;
  alpha_ = params.alpha;
  nndr_ = params.nndr;
  nndr_bf_ = params.nndr_bf;
  ep_dist_ = params.ep_dist;
  conf_prob_ = params.conf_prob;
  min_score_ = params.min_score;
  island_size_ = params.island_size;
  island_offset_ = island_size_ / 2;
  min_inliers_ = params.min_inliers;
  nframes_after_lc_ = params.nframes_after_lc;
  last_lc_result_.status = LC_NOT_DETECTED;
  min_consecutive_loops_ = params.min_consecutive_loops;
  non_sim_candidates_ = params.non_sim_candidates;
  consecutive_loops_ = 0;
  gt_matrix_ = params.gt_matrix;
  debug_loops_ = params.debug_loops;

  geom_params_.b_l_endpts_ = params.b_l_endpts;
  geom_params_.b_l_inters_pts_ = params.b_l_inters_pts;
  geom_params_.b_l_global_rot_ = params.b_l_global_rot;
  geom_params_.b_l_center_pt_ = params.b_l_center_pts;
  geom_params_.ep_dist_ = params.ep_dist;
  geom_params_.conf_prob_ = params.conf_prob;

  num_incorrect_match = 0;
  num_not_found_match = 0;

  
  boost::filesystem::path wrong_matches_fold = params.output_path + std::string("/WrongMatches");
  wrong_matches_path_ = params.output_path + std::string("/WrongMatches/");
  boost::filesystem::remove_all(wrong_matches_fold);
  
  boost::filesystem::path not_found_matches_fold = params.output_path + std::string("/NotFoundMatches");
  not_found_matches_path_ = params.output_path + std::string("/NotFoundMatches/");
  boost::filesystem::remove_all(not_found_matches_fold);

  if (debug_loops_)
  {
    boost::filesystem::create_directory(wrong_matches_fold);
    boost::filesystem::create_directory(not_found_matches_fold);
  }
}

LCDetector::~LCDetector() {}

void LCDetector::process(const unsigned image_id,
                         const std::vector<cv::KeyPoint> &kps,
                         const cv::Mat &descs,
                         LCDetectorResult *result)
{
  result->query_id = image_id;

  // Storing the keypoints and descriptors
  prev_kps_.push_back(kps);
  prev_descs_.push_back(descs);

  // Adding the current image to the queue to be added in the future
  queue_ids_.push(image_id);

  // Assessing if, at least, p images have arrived
  if (queue_ids_.size() < p_) {
    result->status = LC_NOT_ENOUGH_IMAGES;
    result->train_id = 0;
    result->inliers = 0;
    last_lc_result_.status = LC_NOT_ENOUGH_IMAGES;
    return;
  }

  // Adding new hypothesis
  unsigned newimg_id = queue_ids_.front();
  queue_ids_.pop();

  addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id]);

  // Searching similar images in the index
  // Matching the descriptors agains the current visual words
  std::vector<std::vector<cv::DMatch> > matches_feats;

  // Searching the query descriptors against the features
  index_->searchDescriptors(descs, &matches_feats, 2, 64);

  // Filtering matches according to the ratio test
  std::vector<cv::DMatch> matches;
  filterMatches(matches_feats, &matches);

  std::vector<obindex2::ImageMatch> image_matches;

  // We look for similar images according to the filtered matches found
  index_->searchImages(descs, matches, &image_matches, true);

  // Filtering the resulting image matchings
  std::vector<obindex2::ImageMatch> image_matches_filt;
  filterCandidates(image_matches, &image_matches_filt);

  std::vector<Island> islands;
  buildIslands(image_matches_filt, &islands);

  if (!islands.size())
  {
    // No resulting islands
    result->status = LC_NOT_ENOUGH_ISLANDS;
    result->train_id = 0;
    result->inliers = 0;
    last_lc_result_.status = LC_NOT_ENOUGH_ISLANDS;
    return;
  }

  // std::cout << "Resulting Islands:" << std::endl;
  // for (unsigned i = 0; i < islands.size(); i++) {
  //   std::cout << islands[i].toString();
  // }

  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;
  getPriorIslands(last_lc_island_, islands, &p_islands);
  if (p_islands.size())
  {
    island = p_islands[0];
  }

  bool overlap = island.overlaps(last_lc_island_);
  last_lc_island_ = island;

  // if () {
  //   consecutive_loops_++;
  // } else {
  //   consecutive_loops_ = 1;
  // }

  unsigned best_img = island.img_id;

  // Assessing the loop
  if (consecutive_loops_ > min_consecutive_loops_ && overlap)
  {
    // LOOP can be considered as detected
    result->status = LC_DETECTED;
    result->train_id = best_img;
    result->inliers = 0;
    // Store the last result
    last_lc_result_ = *result;
    consecutive_loops_++;
  }
  else
  {
    // We obtain the image matchings, since we need them for compute F
    std::vector<cv::DMatch> tmatches;
    std::vector<cv::Point2f> tquery;
    std::vector<cv::Point2f> ttrain;
    ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
    convertPoints(kps, prev_kps_[best_img], tmatches, &tquery, &ttrain);
    unsigned inliers = checkEpipolarGeometry(tquery, ttrain);

    if (inliers > min_inliers_)
    {
      // LOOP detected
      result->status = LC_DETECTED;
      result->train_id = best_img;
      result->inliers = inliers;
      // Store the last result
      last_lc_result_ = *result;
      consecutive_loops_++;
    }
    else
    {
      result->status = LC_NOT_ENOUGH_INLIERS;
      result->train_id = best_img;
      result->inliers = inliers;
      last_lc_result_.status = LC_NOT_ENOUGH_INLIERS;
      consecutive_loops_ = 0;
    }
  }
  // else {
  //   result->status = LC_NOT_DETECTED;
  //   last_lc_result_.status = LC_NOT_DETECTED;
  // }
}

void LCDetector::debug(const unsigned image_id,
                       const std::vector<cv::KeyPoint> &kps,
                       const cv::Mat &descs,
                       std::ofstream &out_file)
{
  auto start = std::chrono::steady_clock::now();
  // Storing the keypoints and descriptors
  prev_kps_.push_back(kps);
  prev_descs_.push_back(descs);

  // Adding the current image to the queue to be added in the future
  queue_ids_.push(image_id);

  // Assessing if, at least, p images have arrived
  if (queue_ids_.size() < p_)
  {
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";                                                       // min_id
    out_file << 0 << "\t";                                                       // max_id
    out_file << 0 << "\t";                                                       // img_id
    out_file << 0 << "\t";                                                       // overlap
    out_file << 0 << "\t";                                                       // Inliers
    out_file << index_->numDescriptors() << "\t";                                // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
    out_file << std::endl;
    return;
  }

  // Adding new hypothesis
  unsigned newimg_id = queue_ids_.front();
  queue_ids_.pop();

  addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id]);

  // Searching similar images in the index
  // Matching the descriptors agains the current visual words
  std::vector<std::vector<cv::DMatch>> matches_feats;

  // Searching the query descriptors against the features
  index_->searchDescriptors(descs, &matches_feats, 2, 64);

  // Filtering matches according to the ratio test
  std::vector<cv::DMatch> matches;
  filterMatches(matches_feats, &matches);

  std::vector<obindex2::ImageMatch> image_matches;

  // We look for similar images according to the filtered matches found
  index_->searchImages(descs, matches, &image_matches, true);

  // Filtering the resulting image matchings
  std::vector<obindex2::ImageMatch> image_matches_filt;
  filterCandidates(image_matches, &image_matches_filt);

  std::vector<Island> islands;
  buildIslands(image_matches_filt, &islands);

  if (!islands.size())
  {
    // No resulting islands
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";                                                       // min_id
    out_file << 0 << "\t";                                                       // max_id
    out_file << 0 << "\t";                                                       // img_id
    out_file << 0 << "\t";                                                       // overlap
    out_file << 0 << "\t";                                                       // Inliers
    out_file << index_->numDescriptors() << "\t";                                // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
    out_file << std::endl;
    return;
  }

  // std::cout << "Resulting Islands:" << std::endl;
  // for (unsigned i = 0; i < islands.size(); i++) {
  //   std::cout << islands[i].toString();
  // }

  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;
  getPriorIslands(last_lc_island_, islands, &p_islands);
  if (p_islands.size())
  {
    island = p_islands[0];
  }

  bool overlap = island.overlaps(last_lc_island_);
  last_lc_island_ = island;

  unsigned best_img = island.img_id;

  // We obtain the image matchings, since we need them for compute F
  std::vector<cv::DMatch> tmatches;
  std::vector<cv::Point2f> tquery;
  std::vector<cv::Point2f> ttrain;
  ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
  convertPoints(kps, prev_kps_[best_img], tmatches, &tquery, &ttrain);
  unsigned inliers = checkEpipolarGeometry(tquery, ttrain);

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;

  // Writing results
  out_file << island.min_img_id << "\t";                                       // min_id
  out_file << island.max_img_id << "\t";                                       // max_id
  out_file << best_img << "\t";                                                // img_id
  out_file << overlap << "\t";                                                 // overlap
  out_file << inliers << "\t";                                                 // Inliers
  out_file << index_->numDescriptors() << "\t";                                // Voc. Size
  out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
  out_file << std::endl;
}

// Debug IboW function adapted to Multiple Features with custom geometric constraints
void LCDetector::debug(const unsigned image_id,
                       const std::vector<cv::Mat> &v_images,
                       const std::vector<cv::KeyPoint> &kps,
                       const cv::Mat &descs,
                       const std::vector<cv::line_descriptor::KeyLine> &kls,
                       const cv::Mat &descs_l,
                       std::ofstream &out_file)
{
  // std::vector<cv::KeyPoint> kps_l;
  auto start = std::chrono::steady_clock::now();
  // Storing the keypoints and descriptors
  prev_kps_.push_back(kps);
  prev_descs_.push_back(descs);

  // Storing the line keypoints and line descriptors
  // prev_kps_l_.push_back(kps_l);
  prev_kls_.push_back(kls);
  prev_descs_l_.push_back(descs_l);

  // Adding the current image to the queue to be added in the future
  queue_ids_.push(image_id);

  // Assessing if, at least, p images have arrived
  if (queue_ids_.size() < p_)
  {
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";                                                       // min_id
    out_file << 0 << "\t";                                                       // max_id
    out_file << 0 << "\t";                                                       // img_id
    out_file << 0 << "\t";                                                       // overlap
    out_file << 0 << "\t";                                                       // Inliers
    out_file << index_->numDescriptors() + index_l_->numDescriptors() << "\t";                                // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
    out_file << std::endl;
    return;
  }

  // Adding new hypothesis
  unsigned newimg_id = queue_ids_.front();
  queue_ids_.pop();

  // Kls is empty because is not used in the Inv Index
  std::vector<KeyPoint> kls_empty(prev_descs_l_[newimg_id].rows);
  addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id],
           kls_empty, prev_descs_l_[newimg_id]);


  // Searching similar images in the index
  // Matching the descriptors agains the current visual words
  std::vector<std::vector<cv::DMatch>> matches_feats;
  std::vector<std::vector<cv::DMatch>> matches_feats_l;
  // Searching the query descriptors against the features
  index_->searchDescriptors(descs, &matches_feats, 2, 64);
  index_l_->searchDescriptors(descs_l, &matches_feats_l, 2, 64);

  // Filtering matches according to the ratio test
  std::vector<cv::DMatch> matches;
  std::vector<cv::DMatch> matches_l;
  filterMatches(matches_feats, &matches);
  filterMatches(matches_feats_l, &matches_l);

  std::vector<obindex2::ImageMatch> image_matches;
  std::vector<obindex2::ImageMatch> image_matches_l;
  // We look for similar images according to the filtered matches found
  index_->searchImages(descs, matches, &image_matches, true);
  index_l_->searchImages(descs_l, matches_l, &image_matches_l, true);

  // std::cerr << "image_matches[0].score : "   << image_matches[0].score <<   "image_matches_l[0].score : " <<  image_matches_l[0].score << std::endl;

  float non_sim_candidates_p = 1, non_sim_candidates_l = 1;

  if (image_matches[0].score == 0 ||  image_matches_l[0].score == 0)
  {
    if(image_matches[0].score == 0)
    {
      non_sim_candidates_p = 0;
      non_sim_candidates_l = 1;
    }
    else
    {
      non_sim_candidates_p = 1;
      non_sim_candidates_l = 0;
    }
    
  }
  else if (image_matches[0].score >= image_matches_l[0].score)
  {
    non_sim_candidates_p = image_matches_l[0].score / image_matches[0].score;
    non_sim_candidates_l = 1 - non_sim_candidates_p;
  }

  else
  {
    non_sim_candidates_l = image_matches[0].score / image_matches_l[0].score;
    non_sim_candidates_p = 1 - non_sim_candidates_l;
  }

// std::cerr << "non_sim_candidates_p : " << non_sim_candidates_p << "non_sim_candidates_l : " << non_sim_candidates_l << std::endl;
  if (non_sim_candidates_l <0 || non_sim_candidates_l > 1){
  std::cerr << "ERROR: Non_sim_candidates_l : " << non_sim_candidates_l << std::endl;
  std::cerr << " << Press enter to continue" << std::endl;
  std::cin.get();

  }

  if (non_sim_candidates_p < 0 || non_sim_candidates_p > 1)
  {
    std::cerr << "ERROR: Non_sim_candidates_p : " << non_sim_candidates_p << std::endl;

    std::cerr << " << Press enter to continue" << std::endl;
    std::cin.get();
  }

  // Filtering the resulting image matchings
  std::vector<obindex2::ImageMatch> image_matches_filt;
  filterCandidates(image_matches, &image_matches_filt);

  std::vector<obindex2::ImageMatch> image_matches_filt_l;
  filterCandidates(image_matches_l, &image_matches_filt_l);

  // Debug Candidates
  // std::cerr << "------  Feature Image Candidates ------" << std::endl;
  // for (size_t i = 0; i < image_matches_filt.size(); i++)
  // {
  //   std::cerr << ": Image ID " << image_matches_filt[i].image_id << " Score : " << image_matches_filt[i].score << std::endl;
  // }

  // std::cerr << "------  LINE Image Candidates ------" << std::endl;
  // for (size_t i = 0; i < image_matches_filt_l.size(); i++)
  // {
  //   std::cerr << ": Image ID " << image_matches_filt_l[i].image_id << " Score : " << image_matches_filt_l[i].score << std::endl;
  // }

  std::vector<int> image_matches_filt_exist(image_matches_filt.size(), 0);
  // Option B: Alpha Beta filter on resulting vectors
  std::vector<obindex2::ImageMatch> image_matches_to_concat;

  bool fusion_output_measurement = false;
  if (fusion_output_measurement)
  {
    for (size_t i = 0; i < image_matches_filt_l.size(); i++)
    {
      bool found = false;
      for (size_t j = 0; j < image_matches_filt.size(); j++)
      {
        if (image_matches_filt_l.at(i).image_id == image_matches_filt.at(j).image_id)
        {
          image_matches_filt.at(j).score = image_matches_filt_l.at(i).score * alpha_ +
                                           image_matches_filt.at(j).score * (1 - alpha_);
          found = true;
          image_matches_filt_exist.at(j) = 1;
          break;
        }
      }

      //TODO: Add here de Late Fusion
      //Penalize candidates not found in both indexes
      if (!found)
      {
        image_matches_filt_l.at(i).score *= non_sim_candidates_l;
        image_matches_to_concat.push_back(image_matches_filt_l.at(i));
      }
    }

    // Penalize candidates from Kpts not found in both indexes
    for (size_t i = 0; i < image_matches_filt_exist.size(); i++)
    {
      if (image_matches_filt_exist[i] == 0)
      {
        image_matches_filt.at(i).score *= non_sim_candidates_p;
      }
    }
    image_matches_filt.insert(image_matches_filt.end(), image_matches_to_concat.begin(), image_matches_to_concat.end());
  }


  else
  {
    int min_num_matches = image_matches_filt_l.size();
    if (image_matches_filt.size() < image_matches_filt_l.size())
    {
      min_num_matches = image_matches_filt.size();
    }
    if(min_num_matches > 5)
      min_num_matches = 5;
    // Borda Count
    int score = min_num_matches;
    for (size_t i = 0; i < min_num_matches; i++)
    {
      image_matches_filt_l[i].score = score;
      image_matches_filt[i].score = score;

      score --;
    }
    //Sum those that share the same match
    for (size_t i = 0; i < image_matches_filt_l.size(); i++)
    {
      bool found = false;
      for (size_t j = 0; j < image_matches_filt.size(); j++)
      {
        if (image_matches_filt_l.at(i).image_id == image_matches_filt.at(j).image_id)
        {
          image_matches_filt.at(j).score += image_matches_filt_l.at(i).score;
          found = true;
          break;
        }
      }
      if (!found)
      {
        image_matches_to_concat.push_back(image_matches_filt_l.at(i));
      }
    }
    image_matches_filt.insert(image_matches_filt.end(), image_matches_to_concat.begin(), image_matches_to_concat.end());
  }
  
  std::sort(image_matches_filt.begin(), image_matches_filt.end());

  // std::cerr << "------ Combined Feature Image Candidates ------" << std::endl;
  // for (size_t i = 0; i < image_matches_filt.size(); i++)
  // {
  //   std::cerr << ": Image ID " << image_matches_filt[i].image_id << " Score : " << image_matches_filt[i].score << std::endl;
  // }

  std::vector<Island> islands;
  buildIslands(image_matches_filt, &islands);

  if (!islands.size())
  {
    // No resulting islands
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";                                                       // min_id
    out_file << 0 << "\t";                                                       // max_id
    out_file << 0 << "\t";                                                       // img_id
    out_file << 0 << "\t";                                                       // overlap
    out_file << 0 << "\t";                                                       // Inliers
    out_file << index_->numDescriptors() + index_l_->numDescriptors() << "\t";                                // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
    out_file << std::endl;
    return;
  }
  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;
  getPriorIslands(last_lc_island_, islands, &p_islands);
  if (p_islands.size())
  {
    island = p_islands[0];
  }
  bool overlap = island.overlaps(last_lc_island_);
  last_lc_island_ = island;

  unsigned best_img = island.img_id;

  // We obtain the image matchings, since we need them for compute F
  std::vector<cv::DMatch> tmatches(0);
  std::vector<cv::DMatch> tmatches_l(0);

  std::vector<cv::Point2f> tquery;
  std::vector<cv::Point2f> ttrain;
  unsigned inliers;
  if ((descs.cols == 0 || prev_descs_[best_img].cols == 0) &&
          descs_l.cols == 0 ||  prev_descs_l_[best_img].cols == 0)
          {
            inliers = 0;
          }

  else
  {
    if (descs.cols > 0 && prev_descs_[best_img].cols > 0)
      ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);

    if (descs_l.cols > 0 && prev_descs_l_[best_img].cols > 0)
      ratioMatchingBF(descs_l, prev_descs_l_[best_img], &tmatches_l);

    std::unique_ptr<GeomConstr> geomConstraints(std::unique_ptr<GeomConstr>(new GeomConstr(v_images[image_id], v_images[best_img], kps, prev_kps_[best_img], descs, prev_descs_[best_img], tmatches, kls, prev_kls_[best_img], descs_l, prev_descs_l_[best_img], tmatches_l, geom_params_)));

    int pts_inliers = geomConstraints->getPtsInliers();
    int line_inliers = geomConstraints->getLinesInliers();

    //FIXME: Save pts inliers and line inliers for pts and lines separately to plot on Matlab?
    inliers = pts_inliers + line_inliers;
    // }
  }

  bool b_wait = false;

  int display_time = 0;
  if (inliers > min_inliers_ && debug_loops_)
  {
    cv::Mat debug_img;
    if (! DebugProposedIslandWithMatches(v_images, image_id, best_img,
                          inliers, prev_kps_, prev_kls_, tmatches, tmatches_l, display_time, debug_img))
    {
      std::string idx = std::to_string(num_incorrect_match);
      imwrite(wrong_matches_path_ + idx + ".png", debug_img);
      num_incorrect_match++;
    }

    cv::imshow("LC Results", debug_img);
  }

  if(inliers > min_inliers_)
  {
    output_mat_.at<uchar>(image_id, best_img) = 1;
  }

  if (inliers < min_inliers_ && debug_loops_ && gt_matrix_.rows >1)
  {
    cv::Mat gt_row = gt_matrix_.row(image_id);
    double min, max;
    cv::minMaxLoc(gt_row, &min, &max);
    if (max > 0)
    {
      cv::Mat debug_img;
      int displ_time;
      DebugProposedIslandWithMatches(v_images, image_id, best_img,
                          inliers, prev_kps_, prev_kls_, tmatches, tmatches_l, displ_time, debug_img);
      std::string idx = std::to_string(num_not_found_match);
      imwrite(not_found_matches_path_ + idx + ".png", debug_img);
      num_not_found_match++;
    }
  }

  if (b_wait)
  cv::waitKey(0);
  else if( display_time != 0)
  cv::waitKey(10);

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  // Writing results
  out_file << island.min_img_id << "\t";                                       // min_id
  out_file << island.max_img_id << "\t";                                       // max_id
  out_file << best_img << "\t";                                                // img_id
  out_file << overlap << "\t";                                                 // overlap
  out_file << inliers << "\t";                                                 // Inliers
  out_file << index_->numDescriptors() + index_l_->numDescriptors() << "\t";   // Voc. Size
  out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
  out_file << std::endl;
}

void LCDetector::addImage(const unsigned image_id,
                          const std::vector<cv::KeyPoint> &kps,
                          const cv::Mat &descs)
{
  if (index_->numImages() == 0)
  {
    // This is the first image that is inserted into the index
    index_->addImage(image_id, kps, descs);
  }
  else
  {
    // We have to search the descriptor and filter them before adding descs
    // Matching the descriptors
    std::vector<std::vector<cv::DMatch>> matches_feats;

    // Searching the query descriptors against the features
    index_->searchDescriptors(descs, &matches_feats, 2, 64);

    // Filtering matches according to the ratio test
    std::vector<cv::DMatch> matches;
    filterMatches(matches_feats, &matches);

    // Finally, we add the image taking into account the correct matchings
    index_->addImage(image_id, kps, descs, matches);
  }
}

void LCDetector::addImage(const unsigned image_id,
                          const std::vector<cv::KeyPoint> &kps,
                          const cv::Mat &descs,
                          const std::vector<cv::KeyPoint> &kps_l,
                          const cv::Mat &descs_l)
{
  if (index_->numImages() == 0)
  {
    // This is the first image that is inserted into the index
    index_->addImage(image_id, kps, descs);
    std::vector<cv::KeyPoint> v_kps(descs_l.rows);
    index_l_->addImage(image_id, v_kps, descs_l);
  }
  else
  {
    // We have to search the descriptor and filter them before adding descs
    // Matching the descriptors
    std::vector<std::vector<cv::DMatch>> matches_feats;
    std::vector<std::vector<cv::DMatch>> matches_feats_l;

    // Searching the query descriptors against the features
    index_->searchDescriptors(descs, &matches_feats, 2, 64);
    index_l_->searchDescriptors(descs_l, &matches_feats_l, 2, 64);

    // Filtering matches according to the ratio test
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> matches_l;
    filterMatches(matches_feats, &matches);
    filterMatches(matches_feats_l, &matches_l);

    // Finally, we add the image taking into account the correct matchings
    index_->addImage(image_id, kps, descs, matches);

    std::vector<cv::KeyPoint> v_kps(descs_l.rows);
    index_l_->addImage(image_id, v_kps, descs_l, matches_l);
  }
}

void LCDetector::filterMatches(
    const std::vector<std::vector<cv::DMatch>> &matches_feats,
    std::vector<cv::DMatch> *matches)
{
  // Clearing the current matches vector
  matches->clear();

  // Filtering matches according to the ratio test
  for (unsigned m = 0; m < matches_feats.size(); m++)
  {
    if (matches_feats[m][0].distance <= matches_feats[m][1].distance * nndr_)
    {
      matches->push_back(matches_feats[m][0]);
    }
  }
}

void LCDetector::filterCandidates(
    const std::vector<obindex2::ImageMatch> &image_matches,
    std::vector<obindex2::ImageMatch> *image_matches_filt)
{
  image_matches_filt->clear();

  double max_score = image_matches[0].score;
  double min_score = image_matches[image_matches.size() - 1].score;

  for (unsigned i = 0; i < image_matches.size(); i++)
  {
    // Computing the new score
    double new_score = (image_matches[i].score - min_score) /
                       (max_score - min_score);
    // Assessing if this image match is higher than a threshold
    if (new_score > min_score_)
    {
      obindex2::ImageMatch match = image_matches[i];
      match.score = new_score;
      image_matches_filt->push_back(match);
    }
    else
    {
      break;
    }
  }
}

void LCDetector::buildIslands(
    const std::vector<obindex2::ImageMatch> &image_matches,
    std::vector<Island> *islands)
{
  islands->clear();

  // We process each of the resulting image matchings
  for (unsigned i = 0; i < image_matches.size(); i++)
  {
    // Getting information about this match
    unsigned curr_img_id = static_cast<unsigned>(image_matches[i].image_id);
    double curr_score = image_matches[i].score;

    // Theoretical island limits
    unsigned min_id = static_cast<unsigned>(std::max((int)curr_img_id - (int)island_offset_,
                                                     0));
    unsigned max_id = curr_img_id + island_offset_;

    // We search for the closest island
    bool found = false;
    for (unsigned j = 0; j < islands->size(); j++)
    {
      if (islands->at(j).fits(curr_img_id))
      {
        islands->at(j).incrementScore(curr_score);
        found = true;
        islands->at(j).num_votes++;
        break;
      }
      else
      {
        // We adjust the limits of a future island
        islands->at(j).adjustLimits(curr_img_id, &min_id, &max_id);
      }
    }

    // Creating a new island if required
    if (!found)
    {
      Island new_island(curr_img_id,
                        curr_score,
                        min_id,
                        max_id);
      islands->push_back(new_island);
    }
  }

  // Normalizing the final scores according to the number of images
  for (unsigned j = 0; j < islands->size(); j++)
  {

    // islands->at(j).score / islands->at(j).num_votes;
    islands->at(j).normalizeScore();
  }

  std::sort(islands->begin(), islands->end());
}

void LCDetector::getPriorIslands(
    const Island &island,
    const std::vector<Island> &islands,
    std::vector<Island> *p_islands)
{
  p_islands->clear();

  // We search for overlapping islands
  for (unsigned i = 0; i < islands.size(); i++)
  {
    Island tisl = islands[i];
    if (island.overlaps(tisl))
    {
      p_islands->push_back(tisl);
    }
  }
}

unsigned LCDetector::checkEpipolarGeometry(
    const std::vector<cv::Point2f> &query,
    const std::vector<cv::Point2f> &train)
{
  std::vector<uchar> inliers(query.size(), 0);
  if (query.size() > 7)
  {
    cv::Mat F =
        cv::findFundamentalMat(
            cv::Mat(query), cv::Mat(train), // Matching points
            CV_FM_RANSAC,                   // RANSAC method
            ep_dist_,                       // Distance to epipolar line
            conf_prob_,                     // Confidence probability
            inliers);                       // Match status (inlier or outlier)
  }
   // Extract the surviving (inliers) matches
  auto it = inliers.begin();
  unsigned total_inliers = 0;
  for (; it != inliers.end(); it++)
  {
    if (*it)
      total_inliers++;
  }

  return total_inliers;
}

void LCDetector::ratioMatchingBF(const cv::Mat &query,
                                 const cv::Mat &train,
                                 std::vector<cv::DMatch> *matches)
{
  matches->clear();
  cv::BFMatcher matcher(cv::NORM_HAMMING);

  // Matching descriptors
  std::vector<std::vector<cv::DMatch>> matches12;
  matcher.knnMatch(query, train, matches12, 2);

  // Filtering the resulting matchings according to the given ratio
  for (unsigned m = 0; m < matches12.size(); m++)
  {
    if (matches12[m][0].distance <= matches12[m][1].distance * nndr_bf_)
    {
      matches->push_back(matches12[m][0]);
    }
  }
}

void LCDetector::convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                               const std::vector<cv::KeyPoint> &train_kps,
                               const std::vector<cv::DMatch> &matches,
                               std::vector<cv::Point2f> *query,
                               std::vector<cv::Point2f> *train)
{
  query->clear();
  train->clear();
  for (auto it = matches.begin(); it != matches.end(); it++)
  {
    // Get the position of query keypoints
    float x = query_kps[it->queryIdx].pt.x;
    float y = query_kps[it->queryIdx].pt.y;
    query->push_back(cv::Point2f(x, y));

    // Get the position of train keypoints
    x = train_kps[it->trainIdx].pt.x;
    y = train_kps[it->trainIdx].pt.y;
    train->push_back(cv::Point2f(x, y));
  }
}

void LCDetector::convertPoints(const std::vector<cv::KeyPoint> &query_kps,
                               const std::vector<cv::KeyPoint> &train_kps,
                               const std::vector<cv::DMatch> &matches,
                               const std::vector<cv::KeyPoint> &query_kps_l,
                               const std::vector<cv::KeyPoint> &train_kps_l,
                               const std::vector<cv::DMatch> &matches_l,
                               std::vector<cv::Point2f> *query,
                               std::vector<cv::Point2f> *train)
{
  query->clear();
  train->clear();
  for (auto it = matches.begin(); it != matches.end(); it++)
  {
    // Get the position of query keypoints
    float x = query_kps[it->queryIdx].pt.x;
    float y = query_kps[it->queryIdx].pt.y;
    query->push_back(cv::Point2f(x, y));

    // Get the position of train keypoints
    x = train_kps[it->trainIdx].pt.x;
    y = train_kps[it->trainIdx].pt.y;
    train->push_back(cv::Point2f(x, y));
  }

  for (auto it = matches_l.begin(); it != matches_l.end(); it++)
  {
    // Get the position of query keypoints
    float x = query_kps_l[it->queryIdx].pt.x;
    float y = query_kps_l[it->queryIdx].pt.y;
    query->push_back(cv::Point2f(x, y));

    // Get the position of train keypoints
    x = train_kps_l[it->trainIdx].pt.x;
    y = train_kps_l[it->trainIdx].pt.y;
    train->push_back(cv::Point2f(x, y));
  }
}

bool LCDetector::DebugProposedIsland(const std::vector<cv::Mat> &v_images,
                                        const int &query_idx,
                                        const int &train_idx,
                                        const int &score,
                                        int &display_time,
                                        cv::Mat &concat)
{
  bool correct = false;
  //Copy and write the image ID
  cv::Mat query = v_images[query_idx].clone();
  putText(query, std::to_string(query_idx), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0), 2, 0);

  cv::Mat train;
  train = v_images.at(train_idx);
  putText(train, std::to_string(train_idx), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0), 2, 0);

  // Join the two images in one
  cv::Mat black_cols = cv::Mat::zeros(query.rows, 10, CV_8UC3);
  cv::hconcat(query, black_cols, query);
  cv::hconcat(query, train, concat);
  cv::Mat black_rows = cv::Mat::zeros(30, concat.cols, CV_8UC3);
  cv::vconcat(concat, black_rows, concat);
  display_time = 50;

  // Add extra information like if its LC and number of inliers
  putText(concat, "Inliers: " + std::to_string(score), cv::Point(5, concat.rows - 10), 1, 1, cv::Scalar(255, 255, 255), 2, 0);

  if (gt_matrix_.rows > 1)
  {
    if (gt_matrix_.at<uchar>(query_idx, train_idx) == 1)
    {
      correct = true;
      putText(concat, "Correct Loop ", cv::Point(concat.cols / 2, concat.rows - 10), 1, 1, cv::Scalar(0, 255, 0), 2, 0);
      display_time = 300;
    }
    else
    {
      bool loop_found = false;
      for (size_t i = 0; i < 20; i++)
      {
        if (loop_found)
          break;

        for (size_t j = 0; j < 20; j++)
        {
          if ((gt_matrix_.at<uchar>(query_idx + i, train_idx + j) == 1) ||
              (gt_matrix_.at<uchar>(query_idx - i, train_idx - j) == 1) || (gt_matrix_.at<uchar>(query_idx + i, train_idx - j) == 1) ||
              (gt_matrix_.at<uchar>(query_idx + i, train_idx - j) == 1))

            loop_found = true;
        }
      }

      if (loop_found)
      {
        putText(concat, "Lower than 20 Frames to be a Correct Loop ", cv::Point(concat.cols / 2, concat.rows - 10), 1, 1, cv::Scalar(32, 114, 243), 2, 0);
        display_time = 300;
        correct = true;
      }

      else
      {
        putText(concat, "Incorrect Loop ", cv::Point(concat.cols / 2, concat.rows - 10), 1, 1, cv::Scalar(0, 0, 255), 2, 0);

        display_time = 2000;
      }
    }
  }
  else
  {
    putText(concat, "Loop Detected. No GT Available ", cv::Point(concat.cols / 2, concat.rows - 10), 1, 1, cv::Scalar(32, 114, 243), 2, 0);
    display_time = 800;
  }
  return correct;
}

bool LCDetector::DebugProposedIslandWithMatches(const std::vector<cv::Mat> &v_images,
                                        const int &query_idx,
                                        const int &train_idx,
                                        const int &score,
                                        const std::vector<std::vector<cv::KeyPoint> > &v_kps,
                                        const std::vector<std::vector<cv::line_descriptor::KeyLine>> &v_kls,
                                        const std::vector<DMatch> &v_matches,
                                        const std::vector<DMatch> &v_matches_l,
                                        int &display_time,
                                        cv::Mat &matched_img)
{
  bool correct = false;
  //Copy and write the image ID
  cv::Mat query = draw2DLines(v_images[query_idx], v_kls[query_idx]);
  putText(query, std::to_string(query_idx), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0), 2, 0);

  cv::Mat train = draw2DLines(v_images[train_idx], v_kls[train_idx]);

  putText(train, std::to_string(train_idx), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0), 2, 0);

  // Join the two images in one
  // cv::Mat black_cols = cv::Mat::zeros(query.rows, 10, CV_8UC3);
  // cv::hconcat(query, black_cols, query);
  // cv::hconcat(query, train, concat);
  // cv::Mat black_rows = cv::Mat::zeros(30, concat.cols, CV_8UC3);
  // cv::vconcat(concat, black_rows, concat);

 matched_img = DrawLineNPtsMatches(v_images, query_idx, train_idx, v_kls, v_kps,
                      v_matches_l, v_matches);

      // Add extra information like if its LC and number of inliers
      putText(matched_img, "Inliers: " + std::to_string(score), cv::Point(5, matched_img.rows - 10), 1, 1, cv::Scalar(255, 255, 255), 2, 0);

  if (gt_matrix_.rows > 1)
  {
    if (gt_matrix_.at<uchar>(query_idx, train_idx) == 1)
    {
      correct = true;
      putText(matched_img, "Correct Loop ", cv::Point(matched_img.cols / 2, matched_img.rows - 10), 1, 1, cv::Scalar(0, 255, 0), 2, 0);
      display_time = 300;
    }
    else
    {
      bool loop_found = false;
      for (size_t i = 0; i < 20; i++)
      {
        if (loop_found)
          break;

        for (size_t j = 0; j < 20; j++)
        {
          if ((gt_matrix_.at<uchar>(query_idx + i, train_idx + j) == 1) ||
              (gt_matrix_.at<uchar>(query_idx - i, train_idx - j) == 1) || (gt_matrix_.at<uchar>(query_idx + i, train_idx - j) == 1) ||
              (gt_matrix_.at<uchar>(query_idx + i, train_idx - j) == 1))

            loop_found = true;
        }
      }

      if (loop_found)
      {
        putText(matched_img, "Lower than 20 Frames to be a Correct Loop ", cv::Point(matched_img.cols / 2, matched_img.rows - 10), 1, 1, cv::Scalar(32, 114, 243), 2, 0);
        display_time = 300;
        correct = true;
      }

      else
      {
        putText(matched_img, "Incorrect Loop ", cv::Point(matched_img.cols / 2, matched_img.rows - 10), 1, 1, cv::Scalar(0, 0, 255), 2, 0);

        display_time = 2000;
      }
    }
  }
  else
  {
    putText(matched_img, "Loop Detected. No GT Available ", cv::Point(matched_img.cols / 2, matched_img.rows - 10), 1, 1, cv::Scalar(32, 114, 243), 2, 0);
    display_time = 800;
  }
  return correct;
}

cv::Mat LCDetector::DrawLineNPtsMatches(
    std::vector<cv::Mat> v_imgs,
    const int &query_idx,
    const int &train_idx,
    const std::vector<std::vector<KeyLine>> &v_kls,
    const std::vector<std::vector<KeyPoint>> &v_kps,
    const std::vector<DMatch> &kls_matches,
    const std::vector<DMatch> &kpts_matches)
{
  cv::Mat tr_img = v_imgs[train_idx].clone();
  cv::Mat q_img = v_imgs[query_idx].clone();

  std::vector<KeyLine> tr_lines = v_kls[train_idx];
  std::vector<KeyLine> q_lines = v_kls[query_idx];

  std::vector<KeyPoint> tr_kps = v_kps[train_idx];
  std::vector<KeyPoint> q_kps = v_kps[query_idx];

  unsigned int imageWidth = tr_img.cols;
  unsigned int imageHeight = tr_img.rows;

  int lineIDLeft;
  int lineIDRight;
  int lowest1 = 0, highest1 = 255;
  int range1 = (highest1 - lowest1) + 1;
  std::vector<unsigned int> r1l(kls_matches.size()), g1l(kls_matches.size()), b1l(kls_matches.size()); //the color of lines

   std::vector<unsigned int> r1p(kpts_matches.size()), g1p(kpts_matches.size()), b1p(kpts_matches.size()); //the color of lines

  for (unsigned int pair = 0; pair < kls_matches.size(); pair++)
  {
    r1l[pair] = lowest1 + int(rand() % range1);
    g1l[pair] = lowest1 + int(rand() % range1);
    b1l[pair] = 255 - r1l[pair];

    if (kls_matches[pair].trainIdx < 0 || kls_matches[pair].queryIdx < 0)
      continue;

    lineIDLeft = kls_matches[pair].trainIdx;
    lineIDRight = kls_matches[pair].queryIdx;

    cv::Point startPointL = cv::Point(int(tr_lines[lineIDLeft].startPointX), int(tr_lines[lineIDLeft].startPointY));
    cv::Point endPointL = cv::Point(int(tr_lines[lineIDLeft].endPointX), int(tr_lines[lineIDLeft].endPointY));
    cv::line(tr_img, startPointL, endPointL, CV_RGB(r1l[pair], g1l[pair], b1l[pair]), 4, cv::LINE_AA, 0);

    cv::Point startPointR = cvPoint(int(q_lines[lineIDRight].startPointX), int(q_lines[lineIDRight].startPointY));

    cv::Point endPointR = cvPoint(int(q_lines[lineIDRight].endPointX), int(q_lines[lineIDRight].endPointY));

    cv::line(q_img, startPointR, endPointR, CV_RGB(r1l[pair], g1l[pair], b1l[pair]), 4, cv::LINE_AA, 0);
  }

  for (unsigned int pair = 0; pair < kpts_matches.size(); pair++)
  {
    r1p[pair] = lowest1 + int(rand() % range1);
    g1p[pair] = lowest1 + int(rand() % range1);
    b1p[pair] = 255 - r1p[pair];

    if (kpts_matches[pair].trainIdx < 0 || kpts_matches[pair].queryIdx < 0)
      continue;

    lineIDLeft = kpts_matches[pair].trainIdx;
    lineIDRight = kpts_matches[pair].queryIdx;
  
    cv::circle(tr_img,tr_kps[lineIDLeft].pt, 5.0, CV_RGB(r1p[pair], g1p[pair], b1p[pair]),2);

    cv::circle(q_img,q_kps[lineIDRight].pt, 5.0, CV_RGB(r1p[pair], g1p[pair], b1p[pair]),2);
  }

  cv::Mat cvResultColorImage1 = cv::Mat(cv::Size(imageWidth * 2, imageHeight),
                                        tr_img.type(), 3);
  cv::Mat cvResultColorImage2 = cv::Mat(cv::Size(imageWidth * 2, imageHeight),
                                        tr_img.type(), 3);
  cv::Mat cvResultColorImage = cv::Mat(cv::Size(imageWidth * 2, imageHeight),
                                       tr_img.type(), 3);
  cv::Mat roi = cvResultColorImage1(cv::Rect(0, 0, imageWidth, imageHeight));
  cv::resize(tr_img, roi, roi.size(), 0, 0, 0);

  cv::Mat roi2 = cvResultColorImage1(cv::Rect(imageWidth, 0, imageWidth,
                                              imageHeight));
  cv::resize(q_img, roi2, roi2.size(), 0, 0, 0);
  cvResultColorImage1.copyTo(cvResultColorImage2);

  for (unsigned int pair = 0; pair < kls_matches.size(); pair++)
  {
    if (kls_matches[pair].trainIdx < 0 || kls_matches[pair].queryIdx < 0)
      continue;
    lineIDLeft = kls_matches[pair].trainIdx;
    lineIDRight = kls_matches[pair].queryIdx;
    cv::Point startPoint = cv::Point(int(tr_lines[lineIDLeft].startPointX), int(tr_lines[lineIDLeft].startPointY));
    cv::Point endPoint = cv::Point(int(q_lines[lineIDRight].startPointX +
                                       imageWidth),
                                   int(q_lines[lineIDRight].startPointY));
    cv::line(cvResultColorImage2, startPoint, endPoint, CV_RGB(r1l[pair], g1l[pair], b1l[pair]), 1, cv::LINE_AA, 0);
  }

  for (unsigned int pair = 0; pair < kpts_matches.size(); pair++)
  {
    if (kpts_matches[pair].trainIdx < 0 || kpts_matches[pair].queryIdx < 0)
      continue;
    lineIDLeft = kpts_matches[pair].trainIdx;
    lineIDRight = kpts_matches[pair].queryIdx;
    cv::Point startPoint = tr_kps[lineIDLeft].pt;
    cv::Point endPoint = cv::Point(q_kps[lineIDRight].pt.x + imageWidth, q_kps[lineIDRight].pt.y);
    cv::line(cvResultColorImage2, startPoint, endPoint, CV_RGB(r1l[pair], g1l[pair], b1l[pair]), 1, cv::LINE_AA, 0);
  }
  cv::addWeighted(cvResultColorImage1, 0.5, cvResultColorImage2, 0.5, 0.0,
                  cvResultColorImage, -1);

  float scale = 1.4;
  cv::resize(cvResultColorImage, cvResultColorImage, cv::Size(2 * imageWidth * scale, imageHeight * scale));

  return cvResultColorImage;
}

cv::Mat LCDetector::draw2DLines(const cv::Mat gray_img,
                                const std::vector<KeyLine> &keylines)
{
    cv::Mat line_img;
    if (gray_img.type() == CV_8UC1)
    {
        cvtColor(gray_img, line_img, CV_GRAY2RGB);
    }
    else
        line_img = gray_img;

    for (size_t i = 0; i < keylines.size(); i++)
    {
        /* get a random color */
       
        // /* get ends of a line */
        cv::Point pt1 = cv::Point(int(keylines[i].startPointX), int(keylines[i].startPointY));
        cv::Point pt2 = cv::Point(int(keylines[i].endPointX), int(keylines[i].endPointY));

        /* draw line */
        line(line_img, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }

    return line_img;
}

} // namespace ibow_lcd
