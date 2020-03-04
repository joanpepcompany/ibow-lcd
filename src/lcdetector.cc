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

  // Kps is empty because is not used in the Inv Index
  std::vector<KeyPoint> kps_empty(prev_descs_l_[newimg_id].rows);
  addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id],
           kps_empty, prev_descs_l_[newimg_id]);


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
  //FIXME: Use a variable which can control option A or B 
  // WIP
  // Option A: Concatenate two vectors
  // image_matches_filt.insert(image_matches_filt.end(), image_matches_filt_l.begin(), image_matches_filt_l.end());

  // Option B: Alpha Beta filter on resulting vectors
  std::vector<obindex2::ImageMatch> image_matches_to_concat;
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
        break;
      }
    }

    //Penalize candidates not found in both indexes
    if (!found)
    {
      image_matches_filt_l.at(i).score *= non_sim_candidates_;
      image_matches_to_concat.push_back(image_matches_filt_l.at(i));
    }
  }

  image_matches_filt.insert(image_matches_filt.end(), image_matches_to_concat.begin(), image_matches_to_concat.end());

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
  std::vector<cv::DMatch> tmatches;
  std::vector<cv::DMatch> tmatches_l;

  std::vector<cv::Point2f> tquery;
  std::vector<cv::Point2f> ttrain;

  unsigned inliers;
  if (descs.cols == 0 || prev_descs_[best_img].cols == 0)
    inliers = 0;

  else
  {
    ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
    ratioMatchingBF(descs_l, prev_descs_l_[best_img], &tmatches_l);

    if (geom_params_.b_l_center_pt_)
    {
      std::vector<KeyPoint> q_kps_l(descs_l.rows);
      for (size_t j = 0; j < kls.size(); j++)
      {
        q_kps_l.at(j).pt = kls.at(j).pt;
      }

      std::vector<KeyPoint> t_kps_l(prev_descs_l_[best_img].rows);
      for (size_t j = 0; j < prev_kls_[best_img].size(); j++)
      {
        t_kps_l.at(j).pt = prev_kls_[best_img].at(j).pt;
      }
      convertPoints(kps, prev_kps_[best_img], tmatches, q_kps_l, t_kps_l, tmatches_l, &tquery, &ttrain);

      inliers = checkEpipolarGeometry(tquery, ttrain);
    }

    else
    {
      std::unique_ptr<GeomConstr> geomConstraints(std::unique_ptr<GeomConstr>(new GeomConstr(v_images[image_id], v_images[best_img], kps, prev_kps_[best_img], descs, prev_descs_[best_img], tmatches, kls, prev_kls_[best_img], descs_l, prev_descs_l_[best_img], tmatches_l, geom_params_)));

      int pts_inliers = geomConstraints->getPtsInliers();
      int line_inliers = geomConstraints->getLinesInliers();

      // std::cerr << "pts_inliers : " << pts_inliers << std::endl;
      // std::cerr << "line_inliers : " << line_inliers << std::endl;

      //FIXME: Save pts inliers and line inliers for pts and lines separately to plot on Matlab? 
      inliers = pts_inliers + line_inliers;
    }
  }

  bool b_wait = false;

  int display_time = 0;
  if (inliers > 10 && debug_loops_)
  {
    cv::Mat debug_img = DebugProposedIsland(v_images, image_id, best_img, inliers, display_time);

    cv::imshow("LC Results", debug_img);
  }
  
  if (b_wait)
  cv::waitKey(0);
  else if( display_time != 0)
  cv::waitKey(display_time);

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

cv::Mat LCDetector::DebugProposedIsland(const std::vector<cv::Mat> &v_images,
                                        const int &query_idx,
                                        const int &train_idx,
                                        const int &score,
                                        int &display_time)
{
  //Copy and write the image ID
  cv::Mat query = v_images[query_idx].clone();
  putText(query, std::to_string(query_idx), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0), 2, 0);

  cv::Mat train;
  train = v_images.at(train_idx);
  putText(train, std::to_string(train_idx), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0), 2, 0);

  // Join the two images in one
  cv::Mat concat;
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
      putText(concat, "Correct Loop ", cv::Point(concat.cols / 2, concat.rows - 10), 1, 1, cv::Scalar(0, 255, 0), 2, 0);
      display_time = 300;
    }
    else
    {
      bool loop_found = false;
      for (size_t i = 0; i < 5; i++)
      {
        if (loop_found)
          break;

        for (size_t j = 0; j < 5; j++)
        {
          if ((gt_matrix_.at<uchar>(query_idx + i, train_idx + j) == 1) ||
              (gt_matrix_.at<uchar>(query_idx - i, train_idx - j) == 1) || (gt_matrix_.at<uchar>(query_idx + i, train_idx - j) == 1) ||
              (gt_matrix_.at<uchar>(query_idx + i, train_idx - j) == 1))

            loop_found = true;
        }
      }

      if (loop_found)
      {
        putText(concat, "Correct Loop ", cv::Point(concat.cols / 2, concat.rows - 10), 1, 1, cv::Scalar(32, 114, 243), 2, 0);
        display_time = 300;
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
  return concat;
}

} // namespace ibow_lcd
