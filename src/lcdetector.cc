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

namespace ibow_lcd
{

LCDetector::LCDetector(const LCDetectorParams &params) : last_lc_island_(-1, 0.0, -1, -1)
{
  b_use_last_island_ = false;
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
  nndr_bf_lines_ = params.nndr_bf_lines;
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

  logos_params_ = params.logos_params;

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
  if (queue_ids_.size() < p_)
  {
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

  if (inliers > min_inliers_)
  {
    output_mat_.at<uchar>(image_id, best_img) = 1;
  }

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

  
void LCDetector::SearchVocCand(const cv::Mat & descs, std::vector<obindex2::ImageMatch> &image_matches_filt)
{
   // Searching similar images in the index
    // Matching the descriptors agains the current visual words
    std::vector<std::vector<cv::DMatch>> matches_feats;
    // Searching the query descriptors against the features
    index_->searchDescriptors(descs, &matches_feats, 2, 64);
    // Filtering matches according to the ratio test
    std::vector<cv::DMatch> matches;
    filterMatches(matches_feats, &matches);
    std::vector<obindex2::ImageMatch> image_matches;
    index_->searchImages(descs, matches, &image_matches, true);

    std::cerr << "image_matches.size() : " << image_matches.size() << std::endl;

    // Filtering the resulting image matchings
    filterCandidates(image_matches, &image_matches_filt);
    std::cerr << "image_matches_filt.size() : " << image_matches_filt.size() << std::endl;

}

void LCDetector::SearchVocCandLines(const cv::Mat & descs, std::vector<obindex2::ImageMatch> &image_matches_filt)
{
   // Searching similar images in the index
    // Matching the descriptors agains the current visual words
    std::vector<std::vector<cv::DMatch>> matches_feats;
    // Searching the query descriptors against the features
    index_l_->searchDescriptors(descs, &matches_feats, 2, 64);
    // Filtering matches according to the ratio test
    std::vector<cv::DMatch> matches;
    filterMatches(matches_feats, &matches);
    std::vector<obindex2::ImageMatch> image_matches;
    index_l_->searchImages(descs, matches, &image_matches, true);

    std::cerr << "image_matches.size() : " << image_matches.size() << std::endl;

    // Filtering the resulting image matchings
    filterCandidates(image_matches, &image_matches_filt);
    std::cerr << "image_matches_filt.size() : " << image_matches_filt.size() << std::endl;

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
  bool b_multi_thread = false;
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
    out_file << index_->numDescriptors() + index_l_->numDescriptors() << "\t";   // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
    out_file << std::endl;
    return;
  }

  // Adding new hypothesis
  unsigned newimg_id = queue_ids_.front();
  queue_ids_.pop();

  auto st_upd_voc_sthreat_pts = std::chrono::steady_clock::now();
  if (!b_multi_thread)
    addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id]);

  auto diff_upd_voc_sthreat_pts = std::chrono::steady_clock::now() - st_upd_voc_sthreat_pts;
  double upd_voc_sthreat_pts = std::chrono::duration<double, std::milli>(diff_upd_voc_sthreat_pts).count();

  v_time_upd_voc_sthreat_pts_.push_back(upd_voc_sthreat_pts);

  auto st_upd_voc_sthreat_l = std::chrono::steady_clock::now();
  //TODO: Kls is empty because is not used in the Inv Index
  std::vector<KeyPoint> kls_empty(prev_descs_l_[newimg_id].rows);
  if(!b_multi_thread)
    addImageLines(newimg_id, kls_empty, prev_descs_l_[newimg_id]);
  // addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id], kls_empty, prev_descs_l_[newimg_id] );

  auto diff_upd_voc_sthreat_l = std::chrono::steady_clock::now() - st_upd_voc_sthreat_l;
  double upd_voc_sthreat_l = std::chrono::duration<double, std::milli>(diff_upd_voc_sthreat_l).count();

  v_time_upd_voc_sthreat_l_.push_back(upd_voc_sthreat_l);
  
  auto st_upd_voc_multithreat = std::chrono::steady_clock::now();

  if(b_multi_thread)
  {
    auto detect_r = std::async(std::launch::async, &LCDetector::addImages, this, newimg_id, std::ref(prev_kps_[newimg_id]), std::ref(prev_descs_[newimg_id]));

    auto detect_l = std::async(std::launch::async, &LCDetector::addImageLines, this, newimg_id, std::ref(kls_empty), std::ref(prev_descs_l_[newimg_id]));

    detect_r.wait();
    detect_l.wait();
  }

  auto diff_upd_voc_multithreat = std::chrono::steady_clock::now() - st_upd_voc_multithreat;
  double upd_voc_multithreat = std::chrono::duration<double, std::milli>(diff_upd_voc_multithreat).count();

  v_time_upd_voc_multithreat_.push_back(upd_voc_multithreat);


 
  //TODO: Start time
  auto st_search_voc_sthreat_pts = std::chrono::steady_clock::now();

  std::vector<obindex2::ImageMatch> image_matches_filt;
  if (!b_multi_thread)
  {
    SearchVocCand(descs, image_matches_filt);
  }


  auto diff_search_voc_sthreat_pts = std::chrono::steady_clock::now() - st_search_voc_sthreat_pts;
  double search_voc_sthreat_pts = std::chrono::duration<double, std::milli>(diff_search_voc_sthreat_pts).count();

  v_time_search_voc_sthreat_pts_.push_back(search_voc_sthreat_pts);
  //LINES
  auto st_search_voc_sthreat_l = std::chrono::steady_clock::now();

  std::vector<obindex2::ImageMatch> image_matches_filt_l;
   if (!b_multi_thread)
  {
    SearchVocCandLines(descs_l, image_matches_filt_l);
  }
    auto diff_search_voc_sthreat_l = std::chrono::steady_clock::now() - st_search_voc_sthreat_l;
  double search_voc_sthreat_l = std::chrono::duration<double, std::milli>(diff_search_voc_sthreat_l).count();

  v_time_search_voc_sthreat_l_.push_back(search_voc_sthreat_l);

  auto st_search_voc_mthreat = std::chrono::steady_clock::now();
  if (b_multi_thread)
  {
    auto voc_cand_pts = std::async(std::launch::async, &LCDetector::SearchVocCand, this, std::ref(descs), std::ref(image_matches_filt));

    auto voc_cand_l = std::async(std::launch::async, &LCDetector::SearchVocCandLines, this,  std::ref(descs_l), std::ref(image_matches_filt_l));

    voc_cand_pts.wait();
    voc_cand_l.wait();
  }

   auto diff_search_voc_mthreat = std::chrono::steady_clock::now() - st_search_voc_mthreat;
  double search_voc_mthreat = std::chrono::duration<double, std::milli>(diff_search_voc_mthreat).count();

  v_time_search_voc_multithreat_.push_back(search_voc_mthreat);

// TODO: Late Fusion
  std::ofstream output_file;
  cv::FileStorage opencv_file("../results/scores/" + std::to_string(image_id)+ ".yaml", cv::FileStorage::WRITE);
  cv::Mat v_pts_score;
  for (size_t i = 0; i < image_matches_filt.size(); i++)
  {
    v_pts_score.push_back(image_matches_filt[i].score);
  }
  cv::Mat v_lines_score;
  for (size_t i = 0; i < image_matches_filt_l.size(); i++)
  {
    v_lines_score.push_back(image_matches_filt_l[i].score);
  }
 
  opencv_file << "pts_score" << v_pts_score;
  opencv_file << "lines_score" << v_lines_score;

  cv::Mat found_gt = (Mat_<bool>(1,1) << false);
  if(gt_matrix_.rows > 1)
  {
    cv::Mat gt_row = gt_matrix_.row(image_id);
    double min, max;
    cv::minMaxLoc(gt_row, &min, &max);

    if (max > 0)
      found_gt = (Mat_<bool>(1, 1) << true);
  }
 

  opencv_file << "is_LC" << found_gt;
  opencv_file.release();
  
  // std::cerr << std::endl;
  // std::cerr << " << Press enter to continue" << std::endl;
  // std::cin.get();

// END TODO: Late Fusion

  


  // TODO: END TIME
  //  FUSE CANDIDATES OF Visual Vocabulary
  // Borda Count
  auto st_time_merge = std::chrono::steady_clock::now();

  int min_num_matches = image_matches_filt_l.size();
  if (image_matches_filt.size() < image_matches_filt_l.size())
  {
    min_num_matches = image_matches_filt.size();
  }
  // if(min_num_matches > 5)
  //   min_num_matches = 5;

  int score = min_num_matches;
  for (size_t i = 0; i < min_num_matches; i++)
  {
    image_matches_filt_l[i].score = score * image_matches_filt_l[i].score ;
    image_matches_filt[i].score = score * image_matches_filt[i].score;

    score--;
  }
  std::vector<obindex2::ImageMatch> image_matches_to_concat;

  //Sum those that share the same match
  for (size_t i = 0; i < image_matches_filt_l.size(); i++)
  {
    bool found = false;
    for (size_t j = 0; j < image_matches_filt.size(); j++)
    {
      if (image_matches_filt_l.at(i).image_id == image_matches_filt.at(j).image_id)
      {
        image_matches_filt.at(j).score =  sqrt (image_matches_filt.at(j).score  * image_matches_filt_l.at(i).score);
        found = true;
        break;
      }
    }
    if (!found)
    {
      image_matches_filt_l.at(i).score =  image_matches_filt_l.at(i).score*0.5;
      image_matches_to_concat.push_back(image_matches_filt_l.at(i));
    }
  }

  //Divide by 2 those candidates which only are in points list
   for (size_t i = 0; i < image_matches_filt.size(); i++)
  {
    bool found = false;
    for (size_t j = 0; j < image_matches_filt_l.size(); j++)
    {
      if (image_matches_filt.at(i).image_id == image_matches_filt_l.at(j).image_id)
      {
        found = true;
        break;
      }
    }
    if (!found)
    {
      image_matches_filt.at(i).score =  image_matches_filt.at(i).score * 0.5;
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

  // bool borda_islands = false;
  // if (borda_islands)
  // {
  //   std::vector<Island> islands_l;
  //   std::vector<Island> islands_to_concat;

  //   buildIslands(image_matches_filt_l, &islands_l);
  //   if ((islands_l.size() > 0) || (islands.size() > 0))
  //   {
  //     int min_num_matches = islands_l.size();
  //     if (islands.size() < islands_l.size())
  //     {
  //       min_num_matches = islands.size();
  //     }
  //     // if(min_num_matches > 5)
  //     //   min_num_matches = 5;
  //     // Borda Count
  //     int score = min_num_matches;
  //     for (size_t i = 0; i < min_num_matches; i++)
  //     {
  //       islands_l[i].score = score;
  //       islands[i].score = score;

  //       score--;
  //     }
  //     //Sum those that share the same match
  //     for (size_t i = 0; i < islands_l.size(); i++)
  //     {
  //       bool found = false;
  //       for (size_t j = 0; j < islands.size(); j++)
  //       {

  //         if ((islands_l.at(i).img_id >= islands.at(j).min_img_id && islands_l.at(i).img_id <= islands.at(j).max_img_id)||
  //         (islands.at(j).img_id >= islands_l.at(i).min_img_id && islands.at(j).img_id <= islands_l.at(i).max_img_id))
  //         {
  //           islands.at(j).score += islands_l.at(i).score;
  //           found = true;
  //           break;
  //         }
  //       }
  //       if (!found)
  //       {
  //         islands_to_concat.push_back(islands_l.at(i));
  //       }
  //     }
  //     islands.insert(islands.end(), islands_to_concat.begin(), islands_to_concat.end());
  //   }
  // }
  std::sort(islands.begin(), islands.end());

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
    out_file << index_->numDescriptors() + index_l_->numDescriptors() << "\t";   // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t"; // Time
    out_file << std::endl;
    return;
  }
  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;

  bool overlap = false;
  if (b_use_last_island_)
  {
    getPriorIslands(last_lc_island_, islands, &p_islands);
    if (p_islands.size())
    {
      island = p_islands[0];
    }
    bool overlap = island.overlaps(last_lc_island_);
    last_lc_island_ = island;
  }

  
  unsigned best_img = island.img_id;

  auto diff_time_merge = std::chrono::steady_clock::now() - st_time_merge;
  double time_merge_cand = std::chrono::duration<double, std::milli>(diff_time_merge).count();

  v_time_merge_cand_island_select_.push_back(time_merge_cand);
  // TODO: End time mix
  // We obtain the image matchings, since we need them for compute F
  std::vector<cv::DMatch> tmatches(0);
  std::vector<cv::DMatch> tmatches_l(0);

  std::vector<cv::Point2f> tquery;
  std::vector<cv::Point2f> ttrain;
  bool use_ransac = true;
  bool use_logos = false;
  bool use_gms = false;
  unsigned inliers = 0;
  int pts_inliers = 0;
  int line_inliers= 0 ;
  if ((descs.cols == 0 || prev_descs_[best_img].cols == 0) &&
          descs_l.cols == 0 ||
      prev_descs_l_[best_img].cols == 0)
  {
    inliers = 0;
  }

  else if(use_ransac)
  {
    //TODO: start spatial verification time

    auto st_time_spatial_ver = std::chrono::steady_clock::now();

    if (descs.cols > 0 && prev_descs_[best_img].cols > 0)
      ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);

    if (descs_l.cols > 0 && prev_descs_l_[best_img].cols > 0 && geom_params_.b_l_global_rot_)
      ratioMatchingBFLines(kls, prev_kls_[best_img], descs_l, prev_descs_l_[best_img], &tmatches_l);

    else if (descs_l.cols > 0 && prev_descs_l_[best_img].cols > 0 )
      ratioMatchingBFLNoGeom(descs_l, prev_descs_l_[best_img], &tmatches_l);

    //----
    std::unique_ptr<GeomConstr> geomConstraints(std::unique_ptr<GeomConstr>(new GeomConstr(v_images[image_id], v_images[best_img], kps, prev_kps_[best_img], descs, prev_descs_[best_img], tmatches, kls, prev_kls_[best_img], descs_l, prev_descs_l_[best_img], tmatches_l, geom_params_)));

    pts_inliers = geomConstraints->getPtsInliers();
    line_inliers = geomConstraints->getLinesInliers();
    inliers = pts_inliers + line_inliers;
    auto diff_time_spatial_ver = std::chrono::steady_clock::now() - st_time_spatial_ver;
    double time_spatial_ver = std::chrono::duration<double, std::milli>(diff_time_spatial_ver).count();

    v_time_spatial_ver_.push_back(time_spatial_ver);
    v_line_inliers_.push_back(double(line_inliers));
    //TODO: spatial verification time

    //  else inliers= 0;
  }

  else if (use_gms)
  {
    auto st_time_spatial_ver = std::chrono::steady_clock::now();
    
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descs,  prev_descs_[best_img], tmatches);
    MatchingBF(descs_l, prev_descs_l_[best_img], &tmatches_l);
    std::vector<cv::KeyPoint> kpts_kl1;
    std::vector<cv::KeyPoint> kpts_kl2;
    std::vector<cv::DMatch> kls_f_matches;
    PtsNLineComb(kls, prev_kls_[best_img], tmatches_l, tmatches.size(),
                 kpts_kl1, kpts_kl2, kls_f_matches);


    // cv::Mat prev_match_line_img;
    // prev_match_line_img = DrawMatches(kls, prev_kls_[best_img], v_images[image_id], v_images[best_img], tmatches_l);
    // cv::imshow("Output prev to GMS line matches", prev_match_line_img);

    std::vector<cv::KeyPoint> comb_kpts1 = kps;
    std::vector<cv::KeyPoint> comb_kpts2 = prev_kps_[best_img];
    comb_kpts1.insert(comb_kpts1.end(), kpts_kl1.begin(), kpts_kl1.end());
    comb_kpts2.insert(comb_kpts2.end(), kpts_kl2.begin(), kpts_kl2.end());
    std::vector<cv::DMatch> comb_matches = tmatches;
    comb_matches.insert(comb_matches.end(), kls_f_matches.begin(), kls_f_matches.end());

    std::vector<bool> vbInliers;
    gms_matcher gms(comb_kpts1, v_images[image_id].size(), comb_kpts2, v_images[best_img].size(), comb_matches);
    // std::cerr << "Debug comb_matches size" << comb_matches.size() << std::endl;

    int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    // cout << "Get total " << num_inliers << " matches." << endl;

    std::vector<cv::DMatch> matches_gms_kpts;
    std::vector<cv::DMatch> matches_gms_kls;

    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
      if (vbInliers[i] == true)
      {
        if (i < tmatches.size())
        {
          matches_gms_kpts.push_back(comb_matches[i]);
        }

        else
        {
          matches_gms_kls.push_back(comb_matches[i]);
        }
      }
    }

    std::vector<cv::DMatch> filt_l_gms_matches;
    match2LineConv(matches_gms_kls, tmatches_l, tmatches.size(), filt_l_gms_matches);

    // std::cerr << "Number of GMS point matches: " << matches_gms_kpts.size() << " / Number of GMS line matches: " << filt_l_gms_matches.size() << std::endl;

    // cv::Mat match_line_img;
    // match_line_img = DrawMatches(kls, prev_kls_[best_img], v_images[image_id], v_images[best_img], filt_l_gms_matches);

    // cv::imshow("Output line matches", match_line_img);
    // waitKey(0);
    pts_inliers = matches_gms_kpts.size();
    line_inliers = filt_l_gms_matches.size();
    inliers = pts_inliers + line_inliers;

    tmatches = matches_gms_kpts;
    tmatches_l = filt_l_gms_matches;

    auto diff_time_spatial_ver = std::chrono::steady_clock::now() - st_time_spatial_ver;
    double time_spatial_ver = std::chrono::duration<double, std::milli>(diff_time_spatial_ver).count();

    v_time_spatial_ver_.push_back(time_spatial_ver);
    v_line_inliers_.push_back(double(line_inliers));

  }

  else if (use_logos)
  {
     auto st_time_spatial_ver = std::chrono::steady_clock::now();

    Logos *logos = new Logos(&logos_params_);

    if (descs.rows > 0 && prev_descs_[best_img].rows > 0)
     ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
    // Match Keypoints using BF
    if (descs_l.rows > 0 && prev_descs_l_[best_img].rows > 0)
      logos->MatchingBF(descs_l, prev_descs_l_[best_img], &tmatches_l);

    // FIXME: Evaluate if point macthing vector > 0
    std::vector<Pt *> l_pts_1;
    std::vector<Pt *> l_pts_2;
    logos->KptsNLines2PtConv(kps,prev_kps_[best_img] , tmatches,
                             kls, prev_kls_[best_img], tmatches_l,
                             l_pts_1, l_pts_2);
    //Extract matches from LOGOS
    std::vector<cv::DMatch> comb_pt_matches;
    logos->Process(l_pts_1, l_pts_2, comb_pt_matches);

    l_pts_1.clear();
    l_pts_2.clear();
    // Logos to point and line matching conversion
    std::vector<cv::DMatch> kp_pt_matches;
    std::vector<cv::DMatch> line_matches;
    logos->getKptsNLineMatches(comb_pt_matches, tmatches, tmatches_l, kp_pt_matches, line_matches);

    tmatches = kp_pt_matches;
    tmatches_l = line_matches;

    pts_inliers = kp_pt_matches.size();
    line_inliers = line_matches.size();
    inliers = pts_inliers + line_inliers;

    // std::cerr << "pts_inliers : " << pts_inliers << " line_inliers : " << line_inliers << std::endl;
    auto diff_time_spatial_ver = std::chrono::steady_clock::now() - st_time_spatial_ver;
    double time_spatial_ver = std::chrono::duration<double, std::milli>(diff_time_spatial_ver).count();

    v_time_spatial_ver_.push_back(time_spatial_ver);
    v_line_inliers_.push_back(double(line_inliers));
  }

  if( inliers < min_inliers_)
    b_use_last_island_ = false;
  
  else
    b_use_last_island_ = true;

  bool b_wait = false;

  int display_time = 0;
  if (inliers > min_inliers_ && debug_loops_)
  {
    cv::Mat debug_img;
    if (!DebugProposedIslandWithMatches(v_images, image_id, best_img,
                                        pts_inliers, line_inliers, prev_kps_, prev_kls_, tmatches, tmatches_l, display_time, debug_img))
    {
      std::string idx = std::to_string(num_incorrect_match);
      imwrite(wrong_matches_path_ + idx + ".png", debug_img);
      num_incorrect_match++;
    }

    cv::imshow("LC Results", debug_img);
  }

  if (inliers > min_inliers_)
  {
    output_mat_.at<uchar>(image_id, best_img) = 1;
  }

  if (inliers < min_inliers_ && debug_loops_ && gt_matrix_.rows > 1)
  {
    cv::Mat gt_row = gt_matrix_.row(image_id);
    double min, max;
    cv::minMaxLoc(gt_row, &min, &max);
    if (max > 0)
    {
      cv::Mat debug_img;
      int displ_time;
      DebugProposedIslandWithMatches(v_images, image_id, best_img,
                                      pts_inliers, line_inliers, prev_kps_, prev_kls_, tmatches, tmatches_l, displ_time, debug_img);
      std::string idx = std::to_string(num_not_found_match);

      imwrite(not_found_matches_path_ + idx + ".png", debug_img);
      num_not_found_match++;
    }
  }

  if (b_wait)
    cv::waitKey(0);
  else if (display_time != 0)
    cv::waitKey(0);

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

void LCDetector::addImages(const unsigned image_id,
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

void LCDetector::addImageLines(const unsigned image_id,
                               const std::vector<cv::KeyPoint> &kps_l,
                               const cv::Mat &descs_l)
{
  if (index_l_->numImages() == 0)
  {
    // This is the first image that is inserted into the index
    std::vector<cv::KeyPoint> v_kps(descs_l.rows);
    index_l_->addImage(image_id, v_kps, descs_l);
  }
  else
  {
    std::vector<std::vector<cv::DMatch>> matches_feats_l;

    // Searching the query descriptors against the features
    index_l_->searchDescriptors(descs_l, &matches_feats_l, 2, 64);

    // Filtering matches according to the ratio test
    std::vector<cv::DMatch> matches_l;
    filterMatches(matches_feats_l, &matches_l);

    // Finally, we add the image taking into account the correct matchings
    std::vector<cv::KeyPoint> v_kps(descs_l.rows);
    index_l_->addImage(image_id, v_kps, descs_l, matches_l);
  }
}

void LCDetector::filterCandidates(
    const std::vector<obindex2::ImageMatch> &image_matches,
    std::vector<obindex2::ImageMatch> *image_matches_filt)
{
  image_matches_filt->clear();

  double max_score = image_matches[0].score;
  double min_score = image_matches[image_matches.size() - 1].score;
  if(image_matches.size()>1)
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

void LCDetector::MatchingBF(const cv::Mat &query,
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
		matches->push_back(matches12[m][0]);
		// TODO: evaluate if it is necessary to use the second most similar match
		matches->push_back(matches12[m][1]);
		//     }
	}
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

void LCDetector::ratioMatchingBFLNoGeom(const cv::Mat &query,
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
    if (matches12[m][0].distance <= matches12[m][1].distance * nndr_bf_lines_)
    {
      matches->push_back(matches12[m][0]);
    }
  }
}
//TODO: Añadir kls
void LCDetector::ratioMatchingBFLines(const std::vector<cv::line_descriptor::KeyLine> &query_kls, 
const std::vector<cv::line_descriptor::KeyLine> &train_kls, 
const cv::Mat &query,
                                 const cv::Mat &train,
                                 std::vector<cv::DMatch> *matches)
{
  matches->clear();
  cv::BFMatcher matcher(cv::NORM_HAMMING);

  // Matching descriptors
  std::vector<std::vector<cv::DMatch>> matches12;
  matcher.knnMatch(query, train, matches12, 4);

  // Filtering the resulting matchings according to the given ratio
  for (unsigned m = 0; m < matches12.size(); m++)
  {
    bool same_orient_0 = checkOrient(query_kls, train_kls, matches12[m][0]);
    bool same_orient_1 = checkOrient(query_kls, train_kls, matches12[m][1]);
    bool same_orient_2 = checkOrient(query_kls, train_kls, matches12[m][2]);
    if (same_orient_0 && same_orient_1)
    {
      if (matches12[m][0].distance <= matches12[m][1].distance * nndr_bf_lines_)
      {
        matches->push_back(matches12[m][0]);
        continue;
      }
    }
    if (same_orient_0 && !same_orient_1)
    {
      if (same_orient_0 && !same_orient_2)
      {
        matches->push_back(matches12[m][0]);
        continue;
      }
      else if(matches12[m][0].distance <= matches12[m][2].distance * nndr_bf_lines_)
      {
        matches->push_back(matches12[m][0]);
        continue;
      }
    }
    if (same_orient_1 && !same_orient_2)
    {
      if (!(matches12[m][0].distance <= matches12[m][1].distance * nndr_bf_lines_))
      {
        matches->push_back(matches12[m][1]);
        continue;
      }
    }
  }
}

bool LCDetector::checkOrient(const std::vector<cv::line_descriptor::KeyLine> &query_kls, 
const std::vector<cv::line_descriptor::KeyLine> &train_kls, const DMatch &match)
{
  double global_angle = 0.0;
   double angleDif = fabs(query_kls[match.queryIdx].angle + global_angle - train_kls[match.trainIdx].angle) * 57.29;

        if (angleDif > 720)
            angleDif -= 720;

        if (angleDif > 360)
            angleDif -= 360;

        if (!((angleDif > 20) && (angleDif < 160) || (angleDif > 200) && (angleDif < 340)))
        {
            return true;
        }

        else
            return false;
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
                                                const int &score_pts,
                                                const int &score_lines,
                                                const std::vector<std::vector<cv::KeyPoint>> &v_kps,
                                                const std::vector<std::vector<cv::line_descriptor::KeyLine>> &v_kls,
                                                const std::vector<DMatch> &v_matches,
                                                const std::vector<DMatch> &v_matches_l,
                                                int &display_time,
                                                cv::Mat &matched_img)
{
  bool correct = false;
  //Copy and write the image ID
  cv::Mat query;
  if (v_kls[query_idx].size()>0)
    query = draw2DLines(v_images[query_idx], v_kls[query_idx]);

  putText(query, std::to_string(query_idx), cv::Point(20, 20), 1, 1, cv::Scalar(255, 0, 0), 2, 0);
  cv::Mat train;
  if (v_kls[train_idx].size() > 0)
    train = draw2DLines(v_images[train_idx], v_kls[train_idx]);

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
  putText(matched_img, "Inliers pts: " + std::to_string(score_pts), cv::Point(5, matched_img.rows - 30), 1, 1, cv::Scalar(0, 255, 0), 2, 0);

  putText(matched_img, "Inliers lines: " + std::to_string(score_lines), cv::Point(5, matched_img.rows - 10), 1, 1, cv::Scalar(0, 255, 0), 2, 0);

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

  if (!(kls_matches.size()>0 && kpts_matches.size()>0))
  {
    cv::Mat joined_img;
     cv::addWeighted(tr_img, 0.5, q_img, 0.5, 0.0,
                  joined_img, -1);

     return joined_img;
  }

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

    cv::circle(tr_img, tr_kps[lineIDLeft].pt, 5.0, CV_RGB(r1p[pair], g1p[pair], b1p[pair]), 2);

    cv::circle(q_img, q_kps[lineIDRight].pt, 5.0, CV_RGB(r1p[pair], g1p[pair], b1p[pair]), 2);
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

double LCDetector::GlobalRotationImagePair(const std::vector<KeyLine> q_lines, const std::vector<KeyLine> &tr_lines)
{
    double TwoPI = 2 * M_PI;
    double rotationAngle = TwoPI;

    const unsigned int resolution_scale = 20;
    //step 1: compute the angle histogram of lines in the left and right images
    const unsigned int dim = 360 / resolution_scale; //number of the bins of histogram
    unsigned int index;                              //index in the histogram
    double direction;
    double scalar = 180 / (resolution_scale * 3.1415927); //used when compute the index
    double angleShift = (resolution_scale * M_PI) / 360;  //make sure zero is the middle of the interval

    std::array<double, dim> angleHistLeft;
    std::array<double, dim> angleHistRight;
    std::array<double, dim> lengthLeft; //lengthLeft[i] store the total line length of all the lines in the ith angle bin.
    std::array<double, dim> lengthRight;
    angleHistLeft.fill(0);
    angleHistRight.fill(0);
    lengthLeft.fill(0);
    lengthRight.fill(0);

    for (unsigned int linenum = 0; linenum < q_lines.size(); linenum++)
    {
        direction = q_lines[linenum].angle + M_PI + angleShift;
        direction = direction < TwoPI ? direction : (direction - TwoPI);
        index = floor(direction * scalar);
        angleHistLeft[index]++;
        lengthLeft[index] += q_lines[linenum].lineLength;
    }
    for (unsigned int linenum = 0; linenum < tr_lines.size(); linenum++)
    {
        direction = tr_lines[linenum].angle + M_PI + angleShift;
        direction = direction < TwoPI ? direction : (direction - TwoPI);
        index = floor(direction * scalar);
        angleHistRight[index]++;
        lengthRight[index] += tr_lines[linenum].lineLength;
    }
    arrayMultiRatio(angleHistLeft.data(), angleHistLeft.size(), (1 / getNormL2(angleHistLeft.data(), angleHistLeft.size())));
    arrayMultiRatio(angleHistRight.data(), angleHistRight.size(), (1 / getNormL2(angleHistRight.data(), angleHistRight.size())));
    arrayMultiRatio(lengthLeft.data(), lengthLeft.size(), (1 / getNormL2(lengthLeft.data(), lengthLeft.size())));
    arrayMultiRatio(lengthRight.data(), lengthRight.size(), (1 / getNormL2(lengthRight.data(), lengthRight.size())));

    //step 2: find shift to decide the approximate global rotation
    std::array<double, dim> difVec; //the difference vector between left histogram and shifted right histogram
    double minDif = 10;             //the minimal angle histogram difference
    double secondMinDif = 10;       //the second minimal histogram difference
    unsigned int minShift;          //the shift of right angle histogram when minimal difference achieved
    unsigned int secondMinShift;    //the shift of right angle histogram when second minimal difference achieved

    std::array<double, dim> lengthDifVec; //the length difference vector between left and right
    double minLenDif = 10;                //the minimal length difference
    double secondMinLenDif = 10;          //100		  //the second minimal length difference
    unsigned int minLenShift;             //the shift of right length vector when minimal length difference achieved
    unsigned int secondMinLenShift;       //the shift of right length vector when the second minimal length difference achieved

    double normOfVec;
    for (unsigned int shift = 0; shift < dim; shift++)
    {
        for (unsigned int j = 0; j < dim; j++)
        {
            index = j + shift;
            index = index < dim ? index : (index - dim);
            difVec[j] = angleHistLeft[j] - angleHistRight[index];
            lengthDifVec[j] = lengthLeft[j] - lengthRight[index];
        }
        //find the minShift and secondMinShift for angle histogram
        normOfVec = getNormL2(difVec.data(), difVec.size());
        if (normOfVec < secondMinDif)
        {
            if (normOfVec < minDif)
            {
                secondMinDif = minDif;
                secondMinShift = minShift;
                minDif = normOfVec;
                minShift = shift;
            }
            else
            {
                secondMinDif = normOfVec;
                secondMinShift = shift;
            }
        }
        //find the minLenShift and secondMinLenShift of length vector
        normOfVec = getNormL2(lengthDifVec.data(), lengthDifVec.size());
        if (normOfVec < secondMinLenDif)
        {
            if (normOfVec < minLenDif)
            {
                secondMinLenDif = minLenDif;
                secondMinLenShift = minLenShift;
                minLenDif = normOfVec;
                minLenShift = shift;
            }
            else
            {
                secondMinLenDif = normOfVec;
                secondMinLenShift = shift;
            }
        }
    }

    //first check whether there exist an approximate global rotation angle between image pair
    float AcceptableAngleHistogramDifference = 0.49;
    float AcceptableLengthVectorDifference = 0.4;
    if (minDif < AcceptableAngleHistogramDifference && minLenDif < AcceptableLengthVectorDifference)
    {
        rotationAngle = minShift * resolution_scale;
        if (rotationAngle > 90 && 360 - rotationAngle > 90)
        {
            //In most case we believe the rotation angle between two image pairs should belong to [-Pi/2, Pi/2]
            rotationAngle = rotationAngle - 180;
        }
        rotationAngle = rotationAngle * M_PI / 180;
    }
    return rotationAngle;
}

double LCDetector::getNormL2(double *arr, int size)
{
    double result = 0;
    for (int i = 0; i < size; i++)
    {
        result = result + arr[i] * arr[i];
    }
    return sqrt(result);
}
void LCDetector::arrayMultiRatio(double *arr, int size, double ratio)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] * ratio;
    }
}

void LCDetector::PtsNLineComb(const std::vector<KeyLine> &p_kls,
				  const std::vector<KeyLine> &c_kls,
				  const std::vector<cv::DMatch> &matches,
				  const int &kpts_size,
				  std::vector<cv::KeyPoint> &p_kpts,
				  std::vector<cv::KeyPoint> &c_kpts,
				  std::vector<cv::DMatch> &out_matches)
{
	int idx_matches = kpts_size;

	for (size_t i = 0; i < matches.size(); i++)
	{
		// Get correspondence Matched Lines
		KeyLine q_line = p_kls[matches[i].queryIdx];
		KeyLine tr_line = c_kls[matches[i].trainIdx];

		//Fill the current an previous Kpts with the lines information
		cv::KeyPoint p_kpt_st;
		p_kpt_st.angle = q_line.angle;
		p_kpt_st.size = q_line.lineLength; 
		p_kpt_st.pt = q_line.getStartPoint();

		cv::KeyPoint c_kpt_st;
		c_kpt_st.angle = tr_line.angle;
		c_kpt_st.size = tr_line.lineLength; 
		c_kpt_st.pt = tr_line.getStartPoint();

		cv::KeyPoint p_kpt_end;
		p_kpt_end.angle = q_line.angle;
		p_kpt_end.size = q_line.lineLength;
		p_kpt_end.pt = q_line.getEndPoint();

		cv::KeyPoint c_kpt_end;
		c_kpt_end.angle = tr_line.angle;
		c_kpt_end.size = tr_line.lineLength;
		c_kpt_end.pt = tr_line.getEndPoint();

		//Add the kpts to the corresp. vector
		p_kpts.push_back(p_kpt_st);
		p_kpts.push_back(p_kpt_end);

		c_kpts.push_back(c_kpt_st);
		c_kpts.push_back(c_kpt_end);

		//Add matches 1:(prev_pt_st -> curr_pt_st), 2:(prev_pt_st -> curr_pt_end)
		//TODO: add idx_offset when combine with points
		cv::DMatch s_match1(idx_matches, idx_matches, matches[i].distance);
		out_matches.push_back(s_match1);

		cv::DMatch s_match2(idx_matches, idx_matches + 1, matches[i].distance);
		out_matches.push_back(s_match2);

		//Add matches 3:(prev_pt_end -> curr_pt_end), 4:(prev_pt_end -> curr_pt_st)

		cv::DMatch s_match3(idx_matches + 1, idx_matches + 1, matches[i].distance);
		out_matches.push_back(s_match3);

		cv::DMatch s_match4(idx_matches + 1, idx_matches, matches[i].distance);
		out_matches.push_back(s_match4);

		idx_matches += 2;
	}
}

void LCDetector::match2LineConv(const std::vector<cv::DMatch> &gms_matches,
                                const std::vector<cv::DMatch> &bf_matches,
                                const int &kpts_size,
                                std::vector<cv::DMatch> &l_matches)
{	
	std::vector<int> found_match_l(bf_matches.size(), 0);
	for (size_t i = 0; i < gms_matches.size(); i++)
	{
		int idx = gms_matches[i].trainIdx - kpts_size;
		found_match_l[(int)idx / 2]++;		
	}

		for (size_t i = 0; i < found_match_l.size(); i++)
	{
		if (found_match_l[i] > 0)
		{
			l_matches.push_back(bf_matches[i]);
		}
	}
	
}

cv::Mat LCDetector::DrawMatches(const std::vector<KeyLine> &linesInRight, const std::vector<KeyLine> &linesInLeft,const cv::Mat &r_image, const cv::Mat &l_image, const std::vector<DMatch> &matchResult)
{
    cv::Mat rightColorImage = r_image.clone();
    cv::Mat leftColorImage = l_image.clone();
    // cv::Mat leftColorImage, rightColorImage;
    // cvtColor(trainGrayImage, leftColorImage, cv::COLOR_GRAY2RGB);
    // cvtColor(queryGrayImage, rightColorImage, cv::COLOR_GRAY2RGB);
    unsigned int imageWidth = leftColorImage.cols;
    unsigned int imageHeight = leftColorImage.rows;
    double ww1, ww2;
    int lineIDLeft;
    int lineIDRight;
    int lowest1 = 0, highest1 = 255;
    int range1 = (highest1 - lowest1) + 1;
    std::vector<unsigned int> r1(matchResult.size()), g1(matchResult.size()), b1(matchResult.size()); //the color of lines
    for (unsigned int pair = 0; pair < matchResult.size(); pair++)
    {
        r1[pair] = lowest1 + int(rand() % range1);
        g1[pair] = lowest1 + int(rand() % range1);
        b1[pair] = 255 - r1[pair];
        ww1 = 0.2 * (rand() % 5);
        ww2 = 1 - ww1;
        char buf[10];
        sprintf(buf, "%d ", pair);
        if(matchResult[pair].trainIdx < 0 || matchResult[pair].queryIdx < 0)
            continue;
        
        lineIDLeft = matchResult[pair].trainIdx;

        lineIDRight = matchResult[pair].queryIdx;

        cv::Point startPointL = cv::Point(int(linesInLeft[lineIDLeft].startPointX), int(linesInLeft[lineIDLeft].startPointY));
        cv::Point endPointL = cv::Point(int(linesInLeft[lineIDLeft].endPointX), int(linesInLeft[lineIDLeft].endPointY));
        cv::line(leftColorImage, startPointL, endPointL, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, cv::LINE_AA, 0);

        cv::putText(leftColorImage, std::to_string(lineIDLeft), cv::Point(startPointL.x, startPointL.y + 10), 1, 1, Scalar(255, 0, 0), 2, 0);

        cv::Point startPointR = cvPoint(int(linesInRight[lineIDRight].startPointX), int(linesInRight[lineIDRight].startPointY));

        cv::Point endPointR = cvPoint(int(linesInRight[lineIDRight].endPointX), int(linesInRight[lineIDRight].endPointY));

        cv::line(rightColorImage, startPointR, endPointR, CV_RGB(r1[pair], g1[pair], b1[pair]), 4, cv::LINE_AA, 0);
        cv::putText(rightColorImage, std::to_string(lineIDRight), cv::Point(startPointR.x, startPointR.y + 10), 1, 1, Scalar(255, 0, 0), 2, 0);
    }
    
    cv::Mat cvResultColorImage1 = cv::Mat(cv::Size(imageWidth * 2, imageHeight), leftColorImage.type(), 3);
    cv::Mat cvResultColorImage2 = cv::Mat(cv::Size(imageWidth * 2, imageHeight), leftColorImage.type(), 3);
    cv::Mat cvResultColorImage = cv::Mat(cv::Size(imageWidth * 2, imageHeight), leftColorImage.type(), 3);
    cv::Mat roi = cvResultColorImage1(cv::Rect(0, 0, imageWidth, imageHeight));
    cv::resize(leftColorImage, roi, roi.size(), 0, 0, 0);

    cv::Mat roi2 = cvResultColorImage1(cv::Rect(imageWidth, 0, imageWidth, imageHeight));
    cv::resize(rightColorImage, roi2, roi2.size(), 0, 0, 0);
    cvResultColorImage1.copyTo(cvResultColorImage2);

    for (unsigned int pair = 0; pair < matchResult.size(); pair++)
    {
        if(matchResult[pair].trainIdx < 0 || matchResult[pair].queryIdx < 0)
            continue;
        lineIDLeft = matchResult[pair].trainIdx;
        lineIDRight = matchResult[pair].queryIdx;
        cv::Point startPoint = cv::Point(int(linesInLeft[lineIDLeft].startPointX), int(linesInLeft[lineIDLeft].startPointY));
        cv::Point endPoint = cv::Point(int(linesInRight[lineIDRight].startPointX + imageWidth), int(linesInRight[lineIDRight].startPointY));
        cv::line(cvResultColorImage2, startPoint, endPoint, CV_RGB(r1[pair], g1[pair], b1[pair]), 1, cv::LINE_AA, 0);
    }
    cv::addWeighted(cvResultColorImage1, 0.5, cvResultColorImage2, 0.5, 0.0, cvResultColorImage, -1);

    float scale = 1.4;
    cv::resize(cvResultColorImage, cvResultColorImage, cv::Size(2 * imageWidth * scale, imageHeight * scale));
    return cvResultColorImage;
}

} // namespace ibow_lcd
